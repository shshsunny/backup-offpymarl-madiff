REGISTRY = {}

from .rnn_agent import RNNAgent
from .madiff_agent import MADiffAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["madiff"] = MADiffAgent