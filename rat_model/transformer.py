import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Dict, Any, Union
from .block import RATBlock
from .rotary_embedding import RotaryPositionEmbedding
from .utils import RATLogger, validate_model_config, safe_tensor_operation, rat_logger


class RAT(nn.Module):
    """
    RAT: Reinforced Adaptive Transformer

    A next-generation transformer architecture featuring:
    - Adaptive Policy Attention with multiple reinforcement learning policies
    - Rotary Position Embeddings for enhanced positional understanding
    - SwiGLU Feed-Forward networks for better expressiveness
    - Temporal convolution layers for improved sequence modeling
    """

    def __init__(self, vocab_size: int, d_model: int = 768, n_layers: int = 12,
                 n_heads: int = 12, n_policies: int = 3, max_seq_len: int = 2048,
                 dropout: float = 0.1, mlp_ratio: int = 4, use_rope: bool = True,
                 tie_weights: bool = True, use_checkpointing: bool = False):
        super().__init__()

        # Validate configuration
        config = validate_model_config({
            'vocab_size': vocab_size, 'd_model': d_model, 'n_layers': n_layers,
            'n_heads': n_heads, 'n_policies': n_policies, 'max_seq_len': max_seq_len,
            'dropout': dropout, 'mlp_ratio': mlp_ratio, 'use_rope': use_rope,
            'tie_weights': tie_weights, 'use_checkpointing': use_checkpointing
        })

        self.d_model = config['d_model']
        self.vocab_size = config['vocab_size']
        self.max_seq_len = config['max_seq_len']
        self.n_layers = config['n_layers']

        # Token and position embeddings
        self.token_emb = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_emb = nn.Parameter(torch.randn(1, config['max_seq_len'], config['d_model']))

        # Rotary position embeddings
        if config['use_rope']:
            self.rope = RotaryPositionEmbedding(config['d_model'] // config['n_heads'], config['max_seq_len'])
            rat_logger.debug("Rotary Position Embeddings enabled")

        # RAT transformer blocks
        self.blocks = nn.ModuleList([
            RATBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                n_policies=config['n_policies'],
                dropout=config['dropout'],
                use_rope=config['use_rope'],
                mlp_ratio=config['mlp_ratio'],
                use_checkpointing=config['use_checkpointing']
            )
            for _ in range(config['n_layers'])
        ])

        # Output layers
        self.norm = nn.LayerNorm(config['d_model'])
        self.output = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

        # Weight tying for better performance
        if config['tie_weights']:
            self.token_emb.weight = self.output.weight
            rat_logger.debug("Weight tying enabled between embeddings and output layer")

        self.dropout = nn.Dropout(config['dropout'])

        # Initialize weights
        self.apply(self._init_weights)

        rat_logger.info(f"RAT initialized with {config['n_layers']} layers, "
                             f"{config['d_model']} dimensions, {config['n_heads']} heads")
        rat_logger.info(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights using best practices"""
        try:
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
        except Exception as e:
            rat_logger.error(f"Weight initialization failed for {type(module).__name__}: {e}")
            raise

    @safe_tensor_operation
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                past_kvs: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
                use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
        """
        Forward pass through RAT

        Args:
            x: Input token ids [batch_size, seq_len]
            mask: Attention mask [batch_size, seq_len]
            past_kvs: Past key/value caches for generation
            use_cache: Whether to return KV caches for faster generation

        Returns:
            logits or (logits, kv_caches) if use_cache=True
        """
        try:
            # Input validation
            if x.dim() != 2:
                raise ValueError(f"Input must be 2D tensor [batch_size, seq_len], got shape {x.shape}")
            if x.dtype != torch.long:
                rat_logger.warning(f"Input tensor dtype is {x.dtype}, expected torch.long")

            B, L = x.shape

            # Check sequence length
            if L > self.max_seq_len:
                raise ValueError(f"Sequence length {L} exceeds maximum {self.max_seq_len}")

            # Token and position embeddings
            token_emb = self.token_emb(x)  # [B, L, D]
            pos_emb = self.pos_emb[:, :L, :]  # [1, L, D]
            x = self.dropout(token_emb + pos_emb)

            # Initialize KV caches
            if past_kvs is None:
                past_kvs = [None] * len(self.blocks)

            present_kvs = []

            # Process through transformer blocks
            for i, block in enumerate(self.blocks):
                try:
                    # Attach RoPE if available
                    if hasattr(self, 'rope'):
                        block.attention.rope = self.rope

                    # Forward pass through block
                    x, present_kv = block(x, mask=mask, past_kv=past_kvs[i], use_cache=use_cache)
                    present_kvs.append(present_kv)

                except Exception as e:
                    rat_logger.error(f"Error in block {i}: {e}")
                    raise RuntimeError(f"RAT block {i} failed: {e}") from e

            # Final normalization and output projection
            x = self.norm(x)
            logits = self.output(x)

            # Validate output
            expected_shape = (B, L, self.vocab_size)
            if logits.shape != expected_shape:
                raise RuntimeError(f"Output shape mismatch: expected {expected_shape}, got {logits.shape}")

            if use_cache:
                return logits, present_kvs
            return logits

        except Exception as e:
            rat_logger.error(f"RAT forward pass failed: {e}")
            raise

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        memory_mb = total_params * 4 / (1024**2)  # Assuming float32

        return {
            'total_parameters': total_params,
            'memory_mb': memory_mb,
            'd_model': self.d_model,
            'n_layers': self.n_layers,
            'vocab_size': self.vocab_size
        }
