import torch
import torch.nn as nn
from .madiff.helpers import apply_conditioning
from .madiff import utils
from .madiff.utils.arrays import to_torch, to_np
from .madiff.ma_temporal import SharedConvAttentionDeconv
from .madiff.diffusion import GaussianDiffusion
from components.madiff_sequence import SequenceDataset

import numpy as np
from collections import deque
from copy import deepcopy
import einops


class MADiffAgent(nn.Module):
    def __init__(self, input_shape, dataset, args):
        # dataset为SequenceDataset对象，需要用到它的normalizer和mask_generator
        super(MADiffAgent, self).__init__()

        assert isinstance(dataset, SequenceDataset), "dataset must be an instance of SequenceDataset"

        # !========注意：必须使用input_shape（可能包含额外的数据来源id onehot）来指代单个agent的输入维度，而不是observation_dim========!
        self.input_shape = input_shape # 单个agent的输入形状，obs_dim (+ action_dim) (+ n_agents)
        self.args = args

        # gxy的版本中：history_horizon=0, horizon=32
        # 我的版本中：history_horizon=20, horizon=4
        self.model = SharedConvAttentionDeconv(
            n_agents=args.n_agents,
            horizon=args.horizon,
            # horizon=args.horizon + args.history_horizon, # 存疑：改成args.horizon会怎样？
            history_horizon=args.history_horizon,
            transition_dim=input_shape, # observation_dim,
            dim_mults=args.dim_mults,
            returns_condition=args.returns_condition,
            env_ts_condition=False, #args.env_ts_condition,
            dim=args.dim, # 可能是中间层维数
            condition_dropout=args.condition_dropout, # gxy版本：0.1
            residual_attn=args.residual_attn,
            max_path_length=args.max_path_length, # 存疑。gxy版本：32。自己版本由offpymarl实际轨迹长度决定。
                                                  # 需要了解该参数在madiff中有何作用！
            use_temporal_attention=True, # args.use_temporal_attention,
            # device=args.device
        )
        self.diffusion = GaussianDiffusion(
            self.model, 
            n_agents=args.n_agents,
            horizon=args.horizon,
            history_horizon=args.history_horizon,
            observation_dim=input_shape, #observation_dim,
            action_dim=1,  # 存疑：改为args.n_actions if args.agent_output_type == "pi_logits" else 1？
            #动作向量（非onehot）的维数，对于离散动作空间应该是1
            discrete_action=args.discrete_action,
            num_actions=args.n_actions, # getattr(dataset.env, "num_actions", 0), # for discrete action space
            n_timesteps=args.n_diffusion_steps, # gxy版本：爆改为15
            clip_denoised=args.clip_denoised,
            predict_epsilon=args.predict_epsilon,
            hidden_dim=args.hidden_dim, # gxy版本：改为128
            train_only_inv=args.train_only_inv,
            share_inv=args.share_inv,
            joint_inv=False, # args.joint_inv,
            # loss weighting
            action_weight=args.action_weight, # gxy版本：改为128
            loss_weights=args.loss_weights,
            state_loss_weight=None, # args.state_loss_weight,
            opponent_loss_weight=None, # args.opponent_loss_weight,
            loss_discount=args.loss_discount,
            returns_condition=args.returns_condition,
            condition_guidance_w=args.condition_guidance_w,
            data_encoder=utils.IdentityEncoder(), # data_encoder
            use_inv_dyn=True, # args.use_inv_dyn,
            device=args.device,
        )
        
        if getattr(args, "use_ddim_sample", True):
            self.diffusion.set_ddim_scheduler(getattr(args, "n_ddim_steps", 15))

        self.normalizer = dataset.normalizer
        self.mask_generator = dataset.mask_generator
        torch.backends.cudnn.benchmark = True # 第一次forward时会自动选择最优算法，加速运算；适用于batch size固定的情况
        self.queue_init = False

    def init_hidden(self):
        """初始化隐藏状态（如果需要）"""
        # 对于MADiffAgent，不需要隐藏状态，但是需要dequeue记录过去时间步的观测
        self.obs_queue = deque(maxlen=self.args.history_horizon + 1)
        self.queue_init = False
        
        return torch.zeros((1, 1), device=self.args.device) # 无用

    def forward(self, obs, hidden_state=None):
        """
        来自basic_controller的obs：
        (batch_size * n_agents, OBS_DIM = obs_dim + n_agents)
        需要reshape恢复
        """
        # 来自mac的obs是tensor，先转为numpy数组
        obs = obs.cpu().numpy()
        #print("################################")
        #print("obs shape:", obs.shape)
        obs = obs.reshape(-1, self.args.n_agents, self.input_shape)  # (batch_size, n_agents, obs_dim + n_agents)
        #print("obs shape:", obs.shape)
        # TODO：分析并编写出正确利用[model]和[diffusion]进行forward的方法
        # 应当从model 和 diffusion 本身的方法入手，分析它们的输入输出格式
        # 之后再倒推将 batch 中的数据转换为所需格式（有可能需要对不同时间步的batch进行累积和拼接，形成轨迹）
        bs = obs.shape[0]
        if not self.queue_init: # 还未初始化，先做填充
            if self.args.use_zero_padding:
                self.obs_queue.extend([np.zeros_like(obs) for _ in range(self.args.history_horizon)])
            else:
                normed_obs = self.normalizer.normalize(obs, "observations")
                self.obs_queue.extend([normed_obs for _ in range(self.args.history_horizon)])
            self.returns = (
                self.args.test_ret * 
                torch.ones(bs, 1, self.args.n_agents, device=self.args.device)
            ) # 初始化returns：固定值，而不是每次forward之后用update_return_to_go来更新

            self.env_ts = (
                torch.arange(
                    self.args.horizon + self.args.history_horizon, 
                    device=self.args.device
                )
                - self.args.history_horizon
            ) # 相对时间步信息：(-history_horizon, ..., 0, ..., horizon-1)
            self.env_ts = einops.repeat(self.env_ts, "t -> b t", b=bs)
            # 等价操作：self.env_ts = self.env_ts.unsqueeze(0).expand(bs,-1)
            self.queue_init = True # 队列初始化完成

        obs = self.normalizer.normalize(obs, "observations")
        #print("obs shape:", obs.shape)
        #print("normed obs: \n", obs)
        # 怀疑：在obs_dim加入了数据来源id的情况下，会不会把该维度normalize了导致出现异常？

        self.obs_queue.append(obs)
        obs = np.stack(list(self.obs_queue), axis=1)  # 堆叠历史观测
        # obs: (batch_size, history_horizon+1, n_agents, OBS_DIM)

        # 使用模型生成样本

        samples = self._generate_samples(obs, self.returns, self.env_ts)

        # 构造输入给逆动力学模型（输入为(o, o')）
        obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
        obs_comb = obs_comb.reshape(-1, self.args.n_agents, 2 * self.input_shape)

        if self.args.share_inv or self.args.joint_inv:
            if self.args.joint_inv:
                actions = self.diffusion.inv_model(
                    obs_comb.reshape(obs_comb.shape[0], -1)
                ).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)
            else: # share_inv，对所有agents采用同一套inverse dynamics参数
                actions = self.diffusion.inv_model(obs_comb)
                # actions: (batch_size, n_agents, action_dim)
        else:
            actions = torch.stack(
                [
                    self.diffusion.inv_model[i](obs_comb[:, i]) 
                    for i in range(self.args.n_agents)
                ],
                dim=1,
            )
        
        self.env_ts = self.env_ts + 1 # 不能漏掉这个！是作为预测轨迹生成条件的重要信息！！！
        # actions可直接返回，经过avail_actions筛除无效动作后取argmax
        # （具体见basic_controller，注意action_selector的选择）
        actions = actions.reshape(-1, actions.shape[-1]) # 重新合并前两维
        return actions, hidden_state
    
    """def update_return_to_go(self, reward):
        # 由basic_controller调用
        # self.returns: (batch_size, 1, n_agents)
        # reward: (batch_size, 1)，自动广播
        rtg = self.returns
        # (batch_size, 1, n_agents)
        rtg = rtg * self.args.returns_scale
        # reward = torch.tensor(reward, device=rtg.device, dtype=rtg.dtype)# .reshape(1, -1)
        rtg = (rtg - reward) / self.args.discount
        rtg = rtg / self.args.returns_scale
        self.returns = rtg""" # 此方法不应有，而是使用固定的预期return值
    
    """def run_episode(self):

        # 初始化环境和变量
        obs = self.env.reset()  # 重置环境，获取初始观测
        obs = obs[None]  # (1, obs_dim)
        recorded_obs = [deepcopy(obs[:, None])]  # 记录观测
        obs_queue = deque(maxlen=self.args.history_horizon + 1)

        # 初始化历史观测队列
        if self.args.use_zero_padding:
            obs_queue.extend([np.zeros_like(obs) for _ in range(self.args.history_horizon)])
        else:
            normed_obs = self.normalizer.normalize(obs, "observations")
            obs_queue.extend([normed_obs for _ in range(self.args.history_horizon)])

        done = False
        total_reward = 0

        while not done:
            # 归一化当前观测并加入历史队列
            obs = self.normalizer.normalize(obs, "observations")
            obs_queue.append(obs)
            obs = np.stack(list(obs_queue), axis=1)  # 堆叠历史观测

            # 使用模型生成样本
            samples = self._generate_samples(obs)

            # 构造输入给逆动力学模型
            obs_comb = torch.cat([samples[:, 0, :, :], samples[:, 1, :, :]], dim=-1)
            obs_comb = obs_comb.reshape(-1, self.args.n_agents, 2 * self.args.observation_dim)

            if self.args.share_inv or self.args.joint_inv:
                if self.args.joint_inv:
                    actions = self.model.inv_model(
                        obs_comb.reshape(obs_comb.shape[0], -1)
                    ).reshape(obs_comb.shape[0], obs_comb.shape[1], -1)
                else:
                    actions = self.model.inv_model(obs_comb)
            else:
                actions = torch.stack(
                    [self.model.inv_model[i](obs_comb[:, i]) for i in range(self.args.n_agents)],
                    dim=1,
                )

            # 从动作分布中采样动作
            actions = actions.argmax(dim=-1).cpu().numpy()

            # 与环境交互
            next_obs, reward, done, info = self.env.step(actions)
            total_reward += reward

            # 更新观测
            obs = next_obs[None]  # 添加 batch 维度

        return total_reward
    """

    def _generate_samples(self, obs, returns, env_ts):
        # 用扩散模型生成样本
        # obs: (batch_size, history_horizon+1, n_agents, observation_dim)
        # 每个样本对应一个独立的观测序列片段，即history_horizon个过去时间步+当前时间步
        env_ts = env_ts.clone()
        env_ts[torch.where(env_ts < 0)] = self.args.max_path_length
        env_ts[torch.where(env_ts >= self.args.max_path_length)] = self.args.max_path_length

        attention_masks = np.zeros(
            (obs.shape[0], self.args.horizon + self.args.history_horizon, self.args.n_agents, 1)
        )
        attention_masks[:, self.args.history_horizon:] = 1.0

        # 采用centralized execution
        shape = (
            obs.shape[0], 
            self.args.horizon + self.args.history_horizon, 
            *obs.shape[-2:]
        )
        cond_trajectories = np.zeros(shape, dtype=obs.dtype)
        cond_trajectories[:, : self.args.history_horizon + 1] = obs
        agent_mask = np.ones(self.args.n_agents)
        cond_masks = self.mask_generator(shape, agent_mask)  # 全部设置为 1
        conditions = {
            "x": torch.as_tensor(cond_trajectories, device=self.args.device),
            "masks": torch.as_tensor(cond_masks, device=self.args.device),
        }
        attention_masks[:, : self.args.history_horizon] = 1.0
        attention_masks = torch.as_tensor(attention_masks, device=self.args.device)
        
        # 调用扩散模型生成样本
        samples = self.diffusion.conditional_sample(
            conditions,
            returns=returns,  # 如果需要 Return-to-Go，可以传入
            env_ts=None, # env_ts,  # 疑问：是否有影响？
            attention_masks=attention_masks,
        )
        return samples[:, self.args.history_horizon:]  # 去掉历史部分
