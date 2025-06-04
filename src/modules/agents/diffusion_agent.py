import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from .helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
# from utils.utils import Progress, Silent

from .helpers import SinusoidalPosEmb

class MLP(nn.Module):
    """
    MLP Model
    """
    def __init__(self,
                 state_dim,
                 action_dim,
                 device,
                 t_dim=16):

        super(MLP, self).__init__()
        self.device = device

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim), # 将时间步序号进行正弦位置嵌入
            nn.Linear(t_dim, t_dim * 2),
            nn.Mish(),
            nn.Linear(t_dim * 2, t_dim),
        )

        input_dim = state_dim + action_dim + t_dim
        self.mid_layer = nn.Sequential(nn.Linear(input_dim, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish(),
                                       nn.Linear(256, 256),
                                       nn.Mish())

        self.final_layer = nn.Linear(256, action_dim)

    def forward(self, x, time, state):
        if len(time.shape) > 1:
            time = time.squeeze(1)  # added for shaping t from (batch_size, 1) to (batch_size,)
        t = self.time_mlp(time)
        x = torch.cat([x, t, state], dim=1)
        x = self.mid_layer(x)

        return self.final_layer(x)
    
class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, # max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', # clip_denoised=True, 
                 predict_epsilon=True):
        super(Diffusion, self).__init__()
        # max_action和clip_denoised用于在连续动作空间情况下控制动作值范围，这里只有离散动作空间，故去除，输出视为离散动作的logits值
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.max_action = max_action
        self.model = model

        # 模型的beta参数，根据scheduler设置为确定值
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        # self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        # DIY：虽然此处禁用clipping，但有可能以后还会用上，因为生成的值是logits，值过大可能产生数值不稳定问题
        """if self.clip_denoised: 
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()"""

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s): # 执行一步反向采样
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s) # 计算σ_t(x_t | s) 用于去噪
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        # progress = Progress(self.n_timesteps) if verbose else Silent() # 进度条，调试用，这里不取
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            # progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        # progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs): # 采样并返回生成的动作
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action
        # return action.clamp_(-self.max_action, self.max_action)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None): # 一步完成前向加噪过程，x_start为无噪声的样本
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=torch.tensor(1.0)):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) # 执行t步加噪

        x_recon = self.model(x_noisy, t, state) # 从x_noisy中重建x_start或噪声

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=torch.tensor(1.0)):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)


class DiffusionAgent(nn.Module):
    def __init__(self, input_shape, args):


        super(DiffusionAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        # self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.diffusion = Diffusion(
            state_dim=args.rnn_hidden_dim,
            action_dim=args.n_actions,
            model=MLP(state_dim=args.rnn_hidden_dim, action_dim=args.n_actions, device="cpu" if args.buffer_cpu_only else args.device),
            beta_schedule='linear', # args.beta_schedule,
            n_timesteps=5, # args.n_timesteps, # 参考原代码仓库取值
            loss_type='l2', # args.loss_type, # 参考原代码仓库取值
            # predict_epsilon=True # 默认为True
        )

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        # q = self.fc2(h)
        pi_logits = self.diffusion(h) # shape: (batch_size, n_actions)
        return pi_logits, h
    
    def half_forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        return h

    def forward_and_losses(self, obs, act, hidden_state, act_eps=0.05):
        # 传统agent的loss是交叉熵损失，这里采用diffusion agent的去噪损失

        # action为独热编码，先将它转为相应的logits值
        # 真实action占绝大部分概率，其他动作平分剩下的概率
        # act: (batch_size, n_actions)
        batch_size, n_actions = act.shape 
        # logits
        logit1 = 0
        logit0 = np.log(act_eps / (n_actions - act_eps * (n_actions - 1)))
        #print("logit values:", logit1, logit0)
        # 创建logits张量，正确动作位置使用correct_logits，其他位置使用incorrect_logits
        act_logits = torch.full_like(act, logit0, dtype=torch.float32)
        act_logits[act.bool()] = logit1

        # 将obs: (batch_size, obs_dim) 转换为h: (batch_size, rnn_hidden_dim)
        # h作为diffusion model的条件参与训练
        h = self.half_forward(obs, hidden_state)
        # 并非所有loss都有意义（某一步已经终止），因此需要过滤掉无意义的loss
        loss = self.diffusion.loss(act_logits, h)
        #print("loss shape:", loss.shape)
        return h, loss.sum(-1) # loss: (batch_size, )





