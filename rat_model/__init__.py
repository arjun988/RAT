"""
RAT Model Components

A collection of advanced transformer components featuring:
- Adaptive Policy Attention with reinforcement learning
- Rotary Position Embeddings
- SwiGLU Feed-Forward networks
- Temporal convolution layers
"""

from .transformer import RAT
from .attention import AdaptivePolicyAttention
from .block import RATBlock
from .feedforward import SwiGLUFeedForward
from .rotary_embedding import RotaryPositionEmbedding, apply_rotary_pos_emb
from .utils import (
    RATLogger, validate_model_config, safe_tensor_operation,
    ModelCheckpoint, get_optimal_batch_size, rat_logger
)

__version__ = "0.1.0"
__all__ = [
    "RAT",
    "AdaptivePolicyAttention",
    "RATBlock",
    "SwiGLUFeedForward",
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",
    "RATLogger",
    "validate_model_config",
    "safe_tensor_operation",
    "ModelCheckpoint",
    "get_optimal_batch_size",
    "rat_logger"
]
