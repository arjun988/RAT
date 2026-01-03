"""
RAT: Reinforced Adaptive Transformer

A collection of advanced transformer components featuring:
- Adaptive Policy Attention with reinforcement learning
- Rotary Position Embeddings for enhanced positional understanding
- SwiGLU Feed-Forward networks for better expressiveness
- Temporal convolution layers for improved sequence modeling
"""

from ..rat_model import *
from ..trainer import RATTrainer
from ..generation import RATGenerator, gpt_generate
from ..dataset import RATDataset, RATDataLoader, GPTTextDataset, prepare_gpt_dataset

__version__ = "0.1.0"
__all__ = [
    # Core model components
    "RAT",
    "AdaptivePolicyAttention",
    "RATBlock",
    "SwiGLUFeedForward",
    "RotaryPositionEmbedding",
    "apply_rotary_pos_emb",

    # Training and inference
    "RATTrainer",
    "RATGenerator",
    "gpt_generate",

    # Data handling
    "RATDataset",
    "RATDataLoader",
    "GPTTextDataset",
    "prepare_gpt_dataset",

    # Utilities
    "RATLogger",
    "validate_model_config",
    "safe_tensor_operation",
    "ModelCheckpoint",
    "get_optimal_batch_size",
    "rat_logger"
]
