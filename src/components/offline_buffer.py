import os
import h5py
import torch as th
import numpy as np

from .madiff_sequence import SequenceDataset
import modules.agents.madiff.utils as utils
############## DataBatch ##############
class OfflineDataBatch():
    def __init__(self, data, batch_size, max_seq_length, device='cpu') -> None:
        self.data = data
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length # None if taken all length
        self.device = device
        for k, v in self.data.items():
            # (batch_size, T, n_agents, *shape)
            # truncate here, interface directly in offlinebuffer
            self.data[k] = v[:, :max_seq_length].to(self.device)
    
    def __getitem__(self, item):
        if isinstance(item, str):
            if item in self.data:
                return self.data[item]
            elif hasattr(self, item):
                return getattr(self, item)
            else:
                raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))
        else:
            raise ValueError('Cannot index OfflineDataBatch with key "{}"'.format(item))

    def to(self, device=None):
        if device is None:
            device = self.device
        for k, v in self.data.items():
            self.data[k] = v.to(device)
        self.device = device # update self.device
    
    def keys(self):
        return list(self.data.keys())
    
    def assign(self, key, value):
        if key in self.data:
            assert 0, "Cannot assign to existing key"
        self.data[key] = value


############## OfflineBuffer ##############
class OfflineBufferH5(): # One Task
    def __init__(self, args, map_name, quality,
                 data_path='', # deepest folder
                 max_buffer_size=2000,
                 device='cpu',
                 shuffle=True) -> None:
        self.args = args
        self.base_data_folder = args.offline_data_folder
        self.map_name = map_name 
        # map name is an abstract name, can be map_name in sc2/scenario_name in mpe
        self.quality = quality
        # set data_path
        if not isinstance(data_path, list) and os.path.exists(data_path):
            self.data_path_list = [data_path] 
        else:
            self.data_path_list = []
            for i, quality_i in enumerate(quality.split("_")): # e.g. "medium_expert"
                data_path_i = data_path[i] if isinstance(data_path, list) else data_path
                self.data_path_list.append(self.get_data_path(map_name, quality_i, data_path_i))

        self.h5_paths = []
        for final_data_path in self.data_path_list:
            self.h5_paths.extend([os.path.join(final_data_path, f) for f in sorted(os.listdir(final_data_path)) if f.endswith(".h5")])
        # args.logger.console_logger.debug("H5_PATHS: " + str(self.h5_paths))
        #self.h5_paths = [os.path.join(self.data_path, f) for f in os.listdir(self.data_path)]
        print("################H5 PATHS:", self.h5_paths) # debugging
        self.max_buffer_size = 100000000 if max_buffer_size <= 0 else max_buffer_size
        self.device = device # device does not work actually.
        self.shuffle = shuffle
        max_data_size_per_file = self.max_buffer_size//len(self.h5_paths)
        dataset = [self._read_data(self.h5_paths[i], max_data_size_per_file, shuffle) for i in range(len(self.h5_paths))]
        self.data = {
            k: np.concatenate([v[k] for v in dataset], axis=0) for k in dataset[0].keys()
        }
        self.keys = list(self.data.keys())
        self.buffer_size = self.data[self.keys[0]].shape[0]

        if shuffle:
            # shuffle again
            shuffled_idx = np.random.choice(self.buffer_size, self.buffer_size, replace=False)
            self.data = {k: v[shuffled_idx] for k, v in self.data.items()}
        
        """print("+++OfflineBufferH5 init complete")
        print("+++Data scheme:")
        for k, v in self.data.items():
            print(f"{k}: {v.shape}, {v.dtype}, {v.device if isinstance(v, th.Tensor) else 'N/A'}")"""    
        
    
    def get_data_path(self, map_name, quality, data_path):
        data_path = os.path.join(self.base_data_folder, self.args.env, map_name, quality, data_path)
        if all([".h5" not in f for f in os.listdir(data_path)]):
            # automatically find a folder
            existing_folders = [f for f in sorted(os.listdir(data_path)) if os.path.isdir(os.path.join(data_path, f))]
            assert len(existing_folders) > 0
            return os.path.join(data_path, existing_folders[-1])
        else:
            return data_path   

    def _read_data(self, h5_path, max_data_size, shuffle):
        data = {}
        with h5py.File(h5_path, 'r') as f:
            for k in f.keys():
                added_data = f[k][:]
                if k not in data:
                    data[k] = added_data
                else:
                    data[k] = np.concatenate((data[k], added_data), axis=0)
        if not shuffle and data[list(data.keys())[0]].shape[0] > max_data_size:
            data = {k: v[-max_data_size:] for k, v in data.items()}

        keys = list(data.keys())
        original_data_size = data[keys[0]].shape[0]
        data_size = min(original_data_size, max_data_size)

        if shuffle:
            shuffled_idx = np.random.choice(original_data_size, data_size, replace=False)
            data = {k: v[shuffled_idx] for k, v in data.items()}
        return data

    @staticmethod
    def max_t_filled(filled):
        return th.sum(filled, 1).max(0)[0] # filled: (batch_size, T, 1)，计算所有episodes中最大的有效时间步数
    
    def can_sample(self, batch_size):
        return self.buffer_size >= batch_size

    def sample(self, batch_size):
        sampled_ep_idx = np.random.choice(self.buffer_size, batch_size, replace=False)
        
        sampled_data = {k: th.tensor(v[sampled_ep_idx]) for k, v in self.data.items()}
        if self.args.use_corrected_terminated and "corrected_terminated" in sampled_data:
            sampled_data["terminated"] = sampled_data["corrected_terminated"]
        """sampled_data = {}
        for k, v in self.data.items():
            dtype = self.scheme[k].get("dtype", th.float32) if self.scheme is not None and k in self.scheme else th.float32
            sampled_data[k] = th.tensor(v[sampled_ep_idx], dtype=dtype)"""
            
        max_ep_t = self.max_t_filled(filled=sampled_data['filled']).item()
        offline_data_batch = OfflineDataBatch(data=sampled_data, 
                                              batch_size=batch_size, 
                                              max_seq_length=max_ep_t, 
                                              device=self.device)
        return offline_data_batch


class OfflineBuffer():
    def __init__(self, args, map_name, quality,
                data_path='', # deepest folder
                max_buffer_size=2000,
                device='cpu',
                shuffle=True) -> None:

        if args.offline_data_type=="h5":
            self.buffer = OfflineBufferH5(args, map_name, quality,
                                          data_path, max_buffer_size,
                                          device, shuffle)
            self.buffer_size = self.buffer.buffer_size
        else:
            raise NotImplementedError("Do not support offline data type: {}".format(args.offline_data_type))
    
    def can_sample(self, batch_size):
        return self.buffer.can_sample(batch_size)

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def sequential_iter(self, batch_size):
        return self.buffer.sequential_iter(batch_size)

    def reset_sequential_iter(self):
        self.buffer.reset_sequential_iter()

class DataSaver():
    def __init__(self, save_path, logger=None, max_size=2000) -> None:
        self.save_path = save_path
        self.max_size = max_size
        #self.episode_batch = []
        self.data_batch = []
        self.cur_size = 0
        self.part_cnt = 0
        self.logger = logger
        os.makedirs(save_path, exist_ok=True)
    
    def append(self, data):
        self.data_batch.append(data) # data \in OfflineDataBatch/EpisodeBatch
        self.cur_size += data[list(data.keys())[0]].shape[0]
        #if len(self.episode_batch) >= self.max_size:
        if self.cur_size >= self.max_size:
            self.save_batch()
    
    def save_batch(self):
        #if len(self.data_batch) == 0:
        if self.cur_size == 0:
            return
        keys = list(self.data_batch[0].keys())
        data_dict = {k: [] for k in keys}
        for data in self.data_batch:
            for k in keys:
                if isinstance(data[k], th.Tensor):
                    data_dict[k].append(data[k].numpy())
                else:
                    data_dict[k].append(data[k])
                    
        # concatenate e.g. [(x, T, n_agents, *shape), ...] -> [max_size, T, n_agents, *shape]
        data_dict = {k: np.concatenate(v) for k, v in data_dict.items()}
        save_file = os.path.join(self.save_path, "part_{}.h5".format(self.part_cnt))
        with h5py.File(save_file, 'w') as file:
            for k, v in data_dict.items():
                file.create_dataset(k, data=v, compression='gzip', compression_opts=9)
        if self.logger is not None:
            self.logger.console_logger.info("Save offline buffer to {} with {} episodes".format(save_file, self.cur_size))
        else:
            print("Save offline buffer to {} with {} episodes".format(save_file, self.cur_size))
        self.data_batch.clear()
        self.cur_size = 0
        self.part_cnt += 1
    
    def close(self):
        self.save_batch()


#-------------------------------- DIY

def sequence_dataset(offline_buffer):
    """
    offline_buffer是OfflineBuffer类的实例，参考数据形状：
    actions: (4000, 61, 3, 1), int64
    actions_onehot: (4000, 61, 3, 9), float32
    avail_actions: (4000, 61, 3, 9), int32
    filled: (4000, 61, 1), int64
    obs: (4000, 61, 3, 30), float32
    reward: (4000, 61, 1), float32
    state: (4000, 61, 48), float32
    """
    data = offline_buffer.buffer.data

    actions = data['actions']
    avail_actions = data['avail_actions']
    filled = data['filled']
    obs = data['obs']
    reward = data['reward']
    state = data['state']

    n_episodes = actions.shape[0]
    n_agents = actions.shape[2]

    onehot_matrix = np.eye(n_agents)[np.newaxis, :, :] # (1, n_agents, n_agents)


    for i in range(n_episodes):
        ep_length = filled[i].sum().item()
        episode_data = {}
        onehot = np.broadcast_to(onehot_matrix, (ep_length, n_agents, n_agents))
        # 拼接独热向量到obs中：(ep_length, n_agents, obs_dim + n_agents)
        episode_data["observations"] = np.concatenate([obs[i][:ep_length], onehot], axis=-1)
        episode_data["legal_actions"] = avail_actions[i][:ep_length]
        shape = list(reward[i][:ep_length].shape)
        shape[-1] = n_agents
        episode_data["rewards"] = np.broadcast_to(reward[i][:ep_length], tuple(shape))
        episode_data["actions"] = actions[i][:ep_length].squeeze(-1)
        episode_data["terminals"] = np.zeros(
            (ep_length, n_agents), dtype=bool
        )
        episode_data["terminals"][-1] = True
        yield episode_data


def cycle(dl):
    while True:
        for data in dl:
            yield data

class MADiffOfflineBuffer():
    # 桥接OfflineBuffer（offpymarl类）和SequenceDataset（madiff类），并提供面向offpymarl框架的采样接口
    def __init__(self, args, offline_buffer: OfflineBuffer):
        assert args.agent == 'madiff', "MADiffOfflineBuffer only supports madiff_ctce config"
        args.max_path_length = offline_buffer.buffer.data['filled'].shape[1]
        print("################max_path_length: %d", args.max_path_length)
        self.args = args
        self.dataset = SequenceDataset(
            #env_type=args.env_type,
            #env=args.dataset,
            sequence_dataset(offline_buffer), # 初始化生成器，用于将offline_buffer的数据导入SequenceDataset
            n_agents=args.n_agents,
            horizon=args.horizon,
            history_horizon=args.history_horizon,
            normalizer=args.normalizer,
            #preprocess_fns=args.preprocess_fns,
            max_n_episodes=args.offline_max_buffer_size, # args.max_n_episodes,
            use_padding=args.use_padding,
            use_action=args.use_action,
            discrete_action=args.discrete_action,
            max_path_length=args.max_path_length, # 存疑：作用？
            include_returns=args.returns_condition,
            include_env_ts=False, #args.env_ts_condition,
            returns_scale=args.returns_scale,
            discount=args.discount,
            termination_penalty=args.termination_penalty,
            agent_share_parameters=True, # SharedConvAttentionDeconv此值为True # utils.config.import_class(args.model).agent_share_parameters,
            # use_seed_dataset=args.use_seed_dataset,
            use_inv_dyn=True, #args.use_inv_dyn,
            decentralized_execution=args.decentralized_execution,
            use_zero_padding=args.use_zero_padding,
            agent_condition_type="single" if args.decentralized_execution else "all", #args.agent_condition_type,
            pred_future_padding=args.pred_future_padding,
        )
        self.dataloader = cycle(
            th.utils.data.DataLoader(
                self.dataset,
                batch_size=args.offline_batch_size, # args.batch_size,
                num_workers=0,
                shuffle=True,
                pin_memory=True,
            )
        )
    

    def sample(self, batch_size):
        return next(self.dataloader)
        #return self.buffer.sample(batch_size)