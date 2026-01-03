import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from typing import Optional, Tuple, List, Dict, Any
from .attention import AdaptivePolicyAttention
from .feedforward import SwiGLUFeedForward
from .utils import safe_tensor_operation, rat_logger


class RATBlock(nn.Module):
    """
    RAT Transformer Block

    Combines adaptive policy attention, SwiGLU feed-forward networks,
    and temporal convolution for enhanced sequence modeling.
    """

    def __init__(self, d_model: int, n_heads: int, n_policies: int = 3,
                 dropout: float = 0.1, use_rope: bool = True, mlp_ratio: int = 4,
                 use_checkpointing: bool = False):
        super().__init__()

        if d_model <= 0 or n_heads <= 0:
            raise ValueError(f"Invalid dimensions: d_model={d_model}, n_heads={n_heads}")

        self.use_checkpointing = use_checkpointing
        self.d_model = d_model

        # Normalization layers
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.tconv_norm = nn.LayerNorm(d_model)

        # Core components
        self.attention = AdaptivePolicyAttention(
            d_model, n_heads, n_policies=n_policies,
            dropout=dropout, use_rope=use_rope
        )
        self.ffn = SwiGLUFeedForward(d_model, d_ff=d_model * mlp_ratio, dropout=dropout)

        # Temporal convolution for sequence modeling
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),  # Depthwise conv
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 1),  # Pointwise conv
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.dropout = nn.Dropout(dropout)

        rat_logger.debug(f"RATBlock initialized: d_model={d_model}, "
                               f"n_heads={n_heads}, n_policies={n_policies}")

    @safe_tensor_operation
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through RAT block

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask
            past_kv: Past key/value cache for generation
            use_cache: Whether to use KV caching

        Returns:
            Tuple of (output_tensor, kv_cache)
        """
        try:
            # Handle 2D input (expand to 3D)
            if x.dim() == 2:
                rat_logger.warning("Input tensor is 2D, expanding to 3D. This may indicate an issue upstream.")
                x = x.unsqueeze(-1).expand(-1, -1, self.d_model)

            if x.shape[-1] != self.d_model:
                raise ValueError(f"Input feature dimension {x.shape[-1]} doesn't match block dimension {self.d_model}")

            def attn_block(x_inner: torch.Tensor) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
                """Attention sub-block with residual connection"""
                residual = x_inner
                x_inner = self.attn_norm(x_inner)

                if use_cache:
                    attn_out, present_kv = self.attention(x_inner, mask, past_kv, use_cache)
                else:
                    attn_out = self.attention(x_inner, mask, past_kv, use_cache)
                    present_kv = None

                x_inner = residual + self.dropout(attn_out)
                return x_inner, present_kv

            # Apply attention with optional checkpointing
            if self.use_checkpointing and self.training and not use_cache:
                x, present_kv = checkpoint.checkpoint(attn_block, x, use_reentrant=False)
                rat_logger.debug("Gradient checkpointing applied to attention block")
            else:
                x, present_kv = attn_block(x)

            # Temporal convolution block
            residual = x
            x_temp = self.tconv_norm(x)
            x_temp = self.temporal_conv(x_temp.transpose(1, 2)).transpose(1, 2)
            x = residual + self.dropout(x_temp)

            # Feed-forward block
            residual = x
            x = self.ffn_norm(x)
            x = residual + self.dropout(self.ffn(x))

            return x, present_kv

        except Exception as e:
            rat_logger.error(f"RATBlock forward pass failed: {e}")
            raise
