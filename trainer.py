import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Callable
from pathlib import Path
import time
from rat_model.utils import RATLogger, ModelCheckpoint, get_optimal_batch_size, rat_logger


class RATTrainer:
    """
    Advanced trainer for RAT models with comprehensive logging,
    checkpointing, and optimization strategies.
    """

    def __init__(self, model: nn.Module, tokenizer, lr: float = 5e-4,
                 warmup_steps: int = 1000, max_steps: int = 100000,
                 weight_decay: float = 0.1, grad_clip: float = 1.0,
                 device: Optional[torch.device] = None, accum_steps: int = 1,
                 checkpoint_dir: str = "./checkpoints", log_level: int = 20):
        """
        Initialize NeuroForge trainer

        Args:
            model: RAT model to train
            tokenizer: Tokenizer for text processing
            lr: Learning rate
            warmup_steps: Number of warmup steps
            max_steps: Total training steps
            weight_decay: Weight decay for regularization
            grad_clip: Gradient clipping threshold
            device: Training device
            accum_steps: Gradient accumulation steps
            checkpoint_dir: Directory for saving checkpoints
            log_level: Logging level
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = RATLogger("RATTrainer", level=log_level)
        self.checkpoint_manager = ModelCheckpoint(checkpoint_dir)

        # Validate inputs
        if not hasattr(tokenizer, 'pad_token_id'):
            self.logger.warning("Tokenizer missing pad_token_id, setting to 0")
            tokenizer.pad_token_id = 0

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.accum_steps = accum_steps
        self.step_count = 0
        self.max_steps = max_steps
        self.grad_clip = grad_clip

        # Separate parameters for weight decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0
            },
        ]

        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=max_steps,
            pct_start=min(0.1, warmup_steps/max_steps),
            anneal_strategy='cos'
        )

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        # Training statistics
        self.training_stats = {
            'total_steps': 0,
            'total_tokens': 0,
            'best_loss': float('inf'),
            'start_time': time.time()
        }

        self.logger.log_model_info(model, "RAT")
        self.logger.info(f"Trainer initialized: lr={lr}, max_steps={max_steps}, "
                        f"device={self.device}, accum_steps={accum_steps}")
    def train_step(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Dict[str, float]:
        """
        Perform one training step

        Args:
            input_seq: Input token sequences [batch_size, seq_len]
            target_seq: Target token sequences [batch_size, seq_len]

        Returns:
            Dictionary with training metrics
        """
        try:
            self.model.train()

            # Move to device and validate
            input_seq = input_seq.to(self.device)
            target_seq = target_seq.to(self.device)

            if input_seq.shape != target_seq.shape:
                raise ValueError(f"Input and target shapes don't match: {input_seq.shape} vs {target_seq.shape}")

            batch_size, seq_len = input_seq.shape
            num_tokens = batch_size * seq_len

            # Forward pass
            start_time = time.time()
            logits = self.model(input_seq)
            forward_time = time.time() - start_time

            # Compute loss
            loss = self.criterion(
                logits.view(-1, self.model.vocab_size),
                target_seq.view(-1)
            )
            loss = loss / self.accum_steps

            # Backward pass
            loss.backward()

            # Gradient accumulation and optimization
            self.step_count += 1
            should_optimize = (self.step_count % self.accum_steps == 0)

            if should_optimize:
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                # Check for gradient issues
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.logger.warning(f"Invalid gradient norm: {grad_norm}. Skipping optimization step.")
                    self.optimizer.zero_grad()
                    return {
                        "loss": float('nan'),
                        "learning_rate": self.scheduler.get_last_lr()[0],
                        "step": self.step_count,
                        "gradient_norm": float('nan')
                    }

                # Optimization step
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

                # Update statistics
                self.training_stats['total_steps'] += 1
                self.training_stats['total_tokens'] += num_tokens * self.accum_steps

            # Compute perplexity
            current_loss = loss.item() * self.accum_steps
            perplexity = float(torch.exp(torch.tensor(current_loss)))

            # Update best loss
            if current_loss < self.training_stats['best_loss']:
                self.training_stats['best_loss'] = current_loss

            # Prepare metrics
            metrics = {
                "loss": current_loss,
                "perplexity": perplexity,
                "learning_rate": self.scheduler.get_last_lr()[0],
                "step": self.step_count,
                "tokens_processed": self.training_stats['total_tokens'],
                "forward_time": forward_time
            }

            if should_optimize:
                metrics["gradient_norm"] = grad_norm.item()

            # Periodic logging
            if self.step_count % 100 == 0:
                self.logger.log_training_step(
                    self.step_count, current_loss,
                    self.scheduler.get_last_lr()[0],
                    perplexity=perplexity
                )

                # Auto-save checkpoint every 1000 steps
                if self.step_count % 1000 == 0:
                    checkpoint_path = self.checkpoint_manager.save(
                        self.model, self.optimizer, self.scheduler,
                        self.step_count, current_loss,
                        additional_info={
                            'training_stats': self.training_stats.copy(),
                            'perplexity': perplexity
                        }
                    )
                    self.logger.info(f"Auto-saved checkpoint: {checkpoint_path}")

            return metrics

        except Exception as e:
            self.logger.error(f"Training step {self.step_count} failed: {e}")
            # Reset gradients on error
            self.optimizer.zero_grad()
            raise RuntimeError(f"RAT training step failed: {e}") from e

    def save_checkpoint(self, filepath: Optional[str] = None) -> str:
        """Save training checkpoint"""
        return self.checkpoint_manager.save(
            self.model, self.optimizer, self.scheduler,
            self.step_count, self.training_stats.get('best_loss', float('inf')),
            additional_info={'training_stats': self.training_stats}
        )

    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint"""
        checkpoint = self.checkpoint_manager.load(filepath, self.model, self.optimizer, self.scheduler)
        self.step_count = checkpoint.get('step', 0)
        self.training_stats.update(checkpoint.get('training_stats', {}))
        self.logger.info(f"Resumed training from step {self.step_count}")

    def get_training_stats(self) -> Dict[str, Any]:
        """Get comprehensive training statistics"""
        elapsed_time = time.time() - self.training_stats['start_time']
        tokens_per_sec = self.training_stats['total_tokens'] / elapsed_time if elapsed_time > 0 else 0

        return {
            **self.training_stats,
            'elapsed_time': elapsed_time,
            'tokens_per_second': tokens_per_sec,
            'current_step': self.step_count,
            'progress_percent': (self.step_count / self.max_steps) * 100
        }
