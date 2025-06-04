REGISTRY = {}

from .rnn_agent import RNNAgent
from .diffusion_agent import DiffusionAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["diffusion"] = DiffusionAgent