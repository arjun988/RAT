"""
RAT: Reinforced Adaptive Transformer

An advanced transformer architecture featuring:
- Adaptive Policy Attention with multiple reinforcement learning policies
- Rotary Position Embeddings for enhanced positional understanding
- SwiGLU Feed-Forward networks for better expressiveness
- Temporal convolution layers for improved sequence modeling

This package provides everything needed to train and deploy RAT models
for advanced language modeling tasks.
"""

from .rat_model import *
from .trainer import RATTrainer
from .generation import RATGenerator, gpt_generate
from .dataset import RATDataset, RATDataLoader, GPTTextDataset, prepare_gpt_dataset

__version__ = "0.1.0"
__author__ = "RAT Team"
__description__ = "Reinforced Adaptive Transformer with Reinforcement Learning-based Attention"

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
