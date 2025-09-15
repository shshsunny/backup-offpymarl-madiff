# ============ imports for Trainer borrowed from MADiff ===========
import copy
import os

import einops
import torch


from modules.agents.madiff.utils.arrays import apply_dict, batch_to_device, to_device, to_np
from modules.agents.madiff.utils.timer import Timer


# ============ imports for learner interface ===========
from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import RMSprop, Adam
import torch.nn.functional as F
from components.standarize_stream import RunningMeanStd


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA:
    """
    empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        args,
        # dataset,
        # ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        # step_start_ema=2000,
        # update_ema_every=10,
        # log_freq=100,
        # sample_freq=1000,
        # save_freq=1000,
        # label_freq=100000,
        # eval_freq=100000,
        # save_parallel=False,
        # n_reference=8,
        # bucket=None,
        train_device="cuda",
        #save_checkpoints=False,

    ):
        super().__init__()
        self.model = diffusion_model
        self.args = args
        #self.ema = EMA(ema_decay)
        #self.ema_model = copy.deepcopy(self.model)
        #self.update_ema_every = update_ema_every
        #self.save_checkpoints = save_checkpoints

        #self.step_start_ema = step_start_ema

        #assert (
        #    eval_freq % save_freq == 0
        #), f"eval_freq must be a multiple of save_freq, but got {eval_freq} and {save_freq} respectively"
        #self.log_freq = log_freq
        #self.sample_freq = sample_freq
        #self.save_freq = save_freq
        # self.label_freq = label_freq
        #self.eval_freq = eval_freq
        #self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        """self.dataset = dataset
        if dataset is not None:
            self.dataloader = cycle(
                torch.utils.data.DataLoader(
                    self.dataset,
                    batch_size=train_batch_size,
                    num_workers=0,
                    shuffle=True,
                    pin_memory=True,
                )
            )
        """
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        # self.bucket = bucket
        #self.n_reference = n_reference

        # self.reset_parameters()
        #self.step = 0

        # self.evaluator = None
        self.device = train_device
        self.grad_norm = torch.tensor(float('nan'))
    """
    # 禁用MADiff原有评估、测试、渲染等接口，都让offpymarl为之代理
    def set_evaluator(self, evaluator):
        self.evaluator = evaluator

    def finish_training(self):
        if self.step % self.save_freq == 0:
            self.save()
        if self.eval_freq > 0 and self.step % self.eval_freq == 0:
            self.evaluate()
        if self.evaluator is not None:
            del self.evaluator

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)"""

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    # 训练机制上的修改：取消MADiff中的epoch和steps_per_epoch，取消自主采样，train接口只提供给定data batch的单步训练操作
    #def train(self, n_train_steps):
    def train(self, batch, t_env):

        loss, infos = self.model.loss(**batch)
        loss = loss / self.gradient_accumulate_every
        loss.backward()
        
        if (t_env+1) % self.gradient_accumulate_every == 0:
            # 注意offpymarl的norm clipping要适当调整！
            self.grad_norm = th.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return loss, self.grad_norm, infos
        """
        if self.step % self.update_ema_every == 0:
            self.step_ema()

        if self.step % self.save_freq == 0:
            self.save()

        if self.eval_freq > 0 and self.step % self.eval_freq == 0:
            self.evaluate()

        if self.step % self.log_freq == 0:
            infos_str = " | ".join(
                [f"{key}: {val:8.4f}" for key, val in infos.items()]
            )
            logger.print(
                f"{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}"
            )
            metrics = {k: v.detach().item() for k, v in infos.items()}
            logger.log(
                step=self.step, loss=loss.detach().item(), **metrics, flush=True
            )

        if self.sample_freq and self.step == 0:
            self.render_reference(self.n_reference)

        if self.sample_freq and self.step % self.sample_freq == 0:
            # 已知model class 为 GaussianDiffusion
            self.inv_render_samples()

        self.step += 1
        """
    """
    def evaluate(self):
        assert (
            self.evaluator is not None
        ), "Method `evaluate` can not be called when `self.evaluator` is None. Set evaluator with `self.set_evaluator` first."
        self.evaluator.evaluate(load_step=self.step)
    
    
    def save(self):

        #saves model and ema to disk;
        #syncs to storage bucket if a bucket is specified


        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.bucket, logger.prefix, "checkpoint")
        os.makedirs(savepath, exist_ok=True)
        if self.save_checkpoints:
            savepath = os.path.join(savepath, f"state_{self.step}.pt")
        else:
            savepath = os.path.join(savepath, "state.pt")
        torch.save(data, savepath)
        logger.print(f"[ utils/training ] Saved model to {savepath}")

    def load(self):

        # loads model and ema from disk


        loadpath = os.path.join(self.bucket, logger.prefix, "checkpoint/state.pt")
        data = torch.load(loadpath)

        self.step = data["step"]
        self.model.load_state_dict(data["model"])
        self.ema_model.load_state_dict(data["ema"])
    """
    # removed all rendering methods

class MADiffLearner: # wrap Trainer to suite the interface of offpymarl
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        # Optimizer已固定为Adam，在Trainer中
        
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        # integrate Trainer

        self.trainer = Trainer( # 此处表达需要修改
            diffusion_model=self.mac.agent.diffusion,
            args=args,
            train_batch_size=args.batch_size,# args.train_batch_size,
            train_lr=args.lr,
            gradient_accumulate_every=args.gradient_accumulate_every,
            train_device=device,  # use cuda if args.use_cuda else cpu
        )
        self.optimiser = self.trainer.optimizer

    def train(self, batch, t_env: int, episode_num: int):
        # 注意：MADiffLearner与MADiffOfflineBuffer配合使用，batch是SequenceDataset生成的dict
        # Get the relevant quantities
        loss, grad_norm, infos = self.trainer.train(batch, t_env)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat(f"loss", loss.item(), t_env)
            self.logger.log_stat(f"grad_norm", grad_norm.item(), t_env)
            for key, value in infos.items():
                if isinstance(value, th.Tensor):
                    value = value.mean()
                self.logger.log_stat(key, value, t_env)
            self.log_stats_t = t_env

    def cuda(self):
        self.mac.cuda()
    
    def save_models(self, path):
        self.mac.save_models(path)
        #for name, param in self.mac.agent.state_dict().items():
        #    print(f"name: {name}, shape: {param.shape}, dtype: {param.dtype}")
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))
    
    def load_models(self, path):
        self.mac.load_models(path)
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
