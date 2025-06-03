from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from copy import deepcopy

class EpisodeRunnerXP:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}


        # DIY
        self.obs_id_len = getattr(args,"obs_id_len", None)
        if self.obs_id_len != None:
            self.obs_id_num = getattr(args, "obs_id_num", None)

        # Log the first run
        self.log_train_stats_t = -1000000

    # DIY: 稍作改造以适配offpymarl
    def setup(self, scheme, groups, preprocess, mac1, mac2):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac1 = mac1
        self.mac2 = mac2

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    # DIY: 稍作改造以适配offpymarl
    #def run(self, mac1=None, mac2=None, test_mode=False, test_mode_1=False, test_mode_2=False, 
    #        negative_reward=False, tm_id=None, iter=None, eps_greedy_t=0, head_id=None, few_shot=False, lipo_xptm_id=None):
    def run(self, test_mode=False, nolog=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac1.init_hidden(batch_size=self.batch_size)
        self.mac2.init_hidden(batch_size=self.batch_size)
    
        while not terminated:
            
            if self.obs_id_len == None:
                obs = self.env.get_obs()
            else:
                obs = self.env.get_obs()
                if self.obs_id_num != None:
                    onehot = np.eye(self.obs_id_len)[self.obs_id_num]
                else:
                    onehot = np.zeros(self.obs_id_len)
                for i in range(len(obs)):
                    obs[i] = np.concatenate([obs[i], onehot], axis=-1)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [obs]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions1 = self.mac1.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions2 = self.mac2.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions = self.merge_actions(actions1, actions2)
            
            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward
        
            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
        

        if self.obs_id_len == None:
            obs = self.env.get_obs()
        else:
            obs = self.env.get_obs()
            if self.obs_id_num != None:
                onehot = np.eye(self.obs_id_len)[self.obs_id_num]
            else:
                onehot = np.zeros(self.obs_id_len)
            for i in range(len(obs)):
                obs[i] = np.concatenate([obs[i], onehot], axis=-1)

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [obs]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions1 = self.mac1.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        actions2 = self.mac2.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        actions = self.merge_actions(actions1, actions2)

        self.batch.update({"actions": actions}, ts=self.t)

        if not test_mode:
            self.t_env += self.t
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = 'test_' if test_mode else ''
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        cur_returns.append(episode_return)

        _cur_stats = _cur_returns = None
        if not nolog:
            if test_mode and len(self.test_returns) == self.args.test_nepisode:
                print("episode_runner_xp logging...")
                _cur_stats = deepcopy(cur_stats)
                _cur_returns = deepcopy(cur_returns)
                self._log(cur_returns, cur_stats, log_prefix)
            elif not test_mode and self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
                self._log(cur_returns, cur_stats, log_prefix)
                if 'offline' not in self.args.run_file:
                    if hasattr(self.mac.action_selector, "epsilon"):
                        self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
                self.log_train_stats_t = self.t_env

        # DIY: 此runner暂仅供测试使用，直接返回统计数据
        if not test_mode:
            return self.batch
        else:
            if _cur_stats != None:
                return _cur_stats, _cur_returns
            else:
                return None


    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()

    def merge_actions(self, actions1, actions2):
        actions = deepcopy(actions2)
        actions[0, : self.args.n_ego] = actions1[0, : self.args.n_ego]
        return actions