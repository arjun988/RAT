"""
RAT Utilities - Logging, error handling, and common functions
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Union
from pathlib import Path
import sys
import traceback
from datetime import datetime


class RATLogger:
    """Advanced logging system for RAT models"""

    def __init__(self, name: str = "RAT", level: int = logging.INFO, log_file: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # File handler if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

        self.logger.info(f"RAT Logger initialized with level {level}")

    def debug(self, message: str, *args, **kwargs):
        """Log debug message"""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message"""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message"""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message"""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message"""
        self.logger.critical(message, *args, **kwargs)

    def log_model_info(self, model, model_name: str = "RAT"):
        """Log comprehensive model information"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.logger.info(f"=== {model_name} Model Summary ===")
        self.logger.info(f"Total Parameters: {total_params:,}")
        self.logger.info(f"Trainable Parameters: {trainable_params:,}")
        self.logger.info(f"Model Size (MB): {total_params * 4 / (1024**2):.2f}")

        # Log architecture details
        if hasattr(model, 'd_model'):
            self.logger.info(f"Model Dimension: {model.d_model}")
        if hasattr(model, 'n_layers'):
            self.logger.info(f"Number of Layers: {model.n_layers}")
        if hasattr(model, 'n_heads'):
            self.logger.info(f"Attention Heads: {model.n_heads}")
        if hasattr(model, 'vocab_size'):
            self.logger.info(f"Vocabulary Size: {model.vocab_size}")

    def log_training_step(self, step: int, loss: float, lr: float, **metrics):
        """Log training step information"""
        msg = f"Step {step:6d} | Loss: {loss:.4f} | LR: {lr:.2e}"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.4f}"
            else:
                msg += f" | {key}: {value}"
        self.logger.info(msg)

    def log_memory_usage(self, device: torch.device):
        """Log current memory usage"""
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**2
            reserved = torch.cuda.memory_reserved(device) / 1024**2
            self.logger.debug(".1f")
        else:
            self.logger.debug("Memory logging only available for CUDA devices")

    def log_generation_stats(self, prompt: str, generated: str, generation_time: float):
        """Log text generation statistics"""
        self.logger.info(f"Generation Time: {generation_time:.3f}s")
        self.logger.debug(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        self.logger.debug(f"Generated: {generated[:100]}{'...' if len(generated) > 100 else ''}")


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and sanitize model configuration"""
    validated_config = config.copy()

    # Required parameters with defaults
    required_params = {
        'vocab_size': lambda x: x > 0,
        'd_model': lambda x: x > 0 and x % 8 == 0,  # Should be divisible by 8 for attention
        'n_layers': lambda x: x > 0,
        'n_heads': lambda x: x > 0 and config.get('d_model', 768) % x == 0,
        'max_seq_len': lambda x: x > 0,
    }

    errors = []
    for param, validator in required_params.items():
        value = validated_config.get(param)
        if value is None:
            errors.append(f"Missing required parameter: {param}")
        elif not validator(value):
            errors.append(f"Invalid value for {param}: {value}")

    if errors:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(errors))

    # Set defaults for optional parameters
    defaults = {
        'dropout': 0.1,
        'mlp_ratio': 4,
        'use_rope': True,
        'tie_weights': True,
        'use_checkpointing': False,
        'n_policies': 3,
    }

    for param, default_value in defaults.items():
        if param not in validated_config:
            validated_config[param] = default_value

    return validated_config


def safe_tensor_operation(func):
    """Decorator for safe tensor operations with error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise MemoryError(f"CUDA out of memory during {func.__name__}. "
                                "Try reducing batch size or model size.") from e
            elif "size mismatch" in str(e).lower():
                raise ValueError(f"Tensor size mismatch in {func.__name__}. "
                               "Check input dimensions.") from e
            else:
                raise RuntimeError(f"Tensor operation failed in {func.__name__}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error in {func.__name__}: {e}") from e
    return wrapper


def get_optimal_batch_size(model, seq_len: int, max_memory_gb: float = 8.0,
                          device: torch.device = None) -> int:
    """Find optimal batch size based on available memory"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type != 'cuda':
        return 4  # Conservative default for CPU

    # Estimate memory per sample (rough approximation)
    vocab_size = getattr(model, 'vocab_size', 50000)
    d_model = getattr(model, 'd_model', 768)

    # Memory estimate: input + output logits + intermediate activations
    mem_per_sample = (seq_len * d_model + seq_len * vocab_size) * 4  # 4 bytes per float32
    mem_per_sample += seq_len * d_model * 4  # Attention intermediates

    available_memory = max_memory_gb * (1024**3)  # Convert GB to bytes
    max_batch_size = int(available_memory * 0.7 / mem_per_sample)  # 70% of available memory

    return max(1, min(max_batch_size, 64))  # Cap at reasonable maximum


class ModelCheckpoint:
    """Advanced checkpoint management for RAT models"""

    def __init__(self, save_dir: str = "./checkpoints", max_checkpoints: int = 5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.logger = RATLogger("RATCheckpoint")

    def save(self, model, optimizer, scheduler, step: int, loss: float,
             additional_info: Dict[str, Any] = None) -> str:
        """Save model checkpoint with comprehensive metadata"""
        checkpoint_path = self.save_dir / f"neuroforge_step_{step:06d}.pt"

        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'loss': loss,
            'timestamp': datetime.now().isoformat(),
            'model_config': getattr(model, '__dict__', {}),
        }

        if additional_info:
            checkpoint.update(additional_info)

        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Clean up old checkpoints
        self._cleanup_old_checkpoints()

        return str(checkpoint_path)

    def load(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
        """Load model checkpoint"""
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space"""
        checkpoints = list(self.save_dir.glob("neuroforge_step_*.pt"))
        if len(checkpoints) > self.max_checkpoints:
            checkpoints.sort(key=lambda x: x.stat().st_mtime)
            for old_ckpt in checkpoints[:-self.max_checkpoints]:
                old_ckpt.unlink()
                self.logger.debug(f"Removed old checkpoint: {old_ckpt}")


# Global logger instance
rat_logger = RATLogger("RAT")
