import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
from .rotary_embedding import apply_rotary_pos_emb
from .utils import safe_tensor_operation, rat_logger


class AdaptivePolicyAttention(nn.Module):
    """
    Adaptive Policy Attention with Reinforcement Learning

    Implements dynamic head gating using multiple RL-based policy networks
    that adaptively control attention head contributions based on input context.
    """

    def __init__(self, d_model: int, n_heads: int, n_policies: int = 3,
                 dropout: float = 0.1, use_rope: bool = True):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_policies = n_policies
        self.use_rope = use_rope

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # Policy networks for adaptive head gating
        self.policy_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_heads),
                nn.Softmax(dim=-1)
            ) for _ in range(n_policies)
        ])

        # Policy selector network
        self.policy_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_policies),
            nn.Softmax(dim=-1)
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

        rat_logger.debug(f"AdaptivePolicyAttention initialized: {n_heads} heads, "
                               f"{n_policies} policies, d_model={d_model}")

    @safe_tensor_operation
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with adaptive policy attention

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len]
            past_kv: Past key/value cache for generation
            use_cache: Whether to cache KV for faster generation

        Returns:
            Attention output or (output, kv_cache) if use_cache=True
        """
        try:
            B, L, D = x.shape

            if D != self.d_model:
                raise ValueError(f"Input dimension {D} doesn't match model dimension {self.d_model}")

            # Linear projections and reshape
            Q = self.w_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, H, L, D_H]
            K = self.w_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.w_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

            # Apply rotary positional embeddings if available
            if hasattr(self, 'rope') and self.use_rope:
                try:
                    Q, K = apply_rotary_pos_emb(Q, K, *self.rope(x, L))
                except Exception as e:
                    rat_logger.warning(f"RoPE application failed, proceeding without: {e}")

            # Concatenate with past KV cache if provided
            if past_kv is not None:
                past_k, past_v = past_kv
                if past_k.shape[1] != self.n_heads or past_v.shape[1] != self.n_heads:
                    raise ValueError("Past KV cache head dimension mismatch")
                K = torch.cat([past_k, K], dim=2)
                V = torch.cat([past_v, V], dim=2)

            seq_len = K.size(2)  # May be longer than L due to past cache

            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, H, L, seq_len]

            # Apply causal mask
            if mask is None:
                causal_mask = torch.triu(torch.ones(L, seq_len, device=x.device), diagonal=1).bool()
                scores = scores.masked_fill(causal_mask[None, None, :, :], -1e9)
            else:
                # Apply custom mask
                if mask.shape != (B, L):
                    rat_logger.warning(f"Mask shape {mask.shape} doesn't match expected {(B, L)}")
                scores = scores.masked_fill(~mask[:, None, :, None].bool(), -1e9)

            # Adaptive policy gating
            try:
                # Get policy weights for this sequence
                policy_weights = self.policy_selector(x.mean(dim=1))  # [B, n_policies]

                # Compute head gates from all policies
                head_gates = torch.zeros(B, self.n_heads, 1, 1, device=x.device)
                for i, policy_net in enumerate(self.policy_nets):
                    policy_gate = policy_net(x.mean(dim=1))  # [B, n_heads]
                    head_gates += policy_weights[:, i:i+1, None, None] * policy_gate.view(B, self.n_heads, 1, 1)

                # Apply adaptive gating to attention scores
                scores = scores * head_gates

            except Exception as e:
                rat_logger.warning(f"Policy gating failed, using uniform attention: {e}")
                # Continue without gating if policy networks fail

            # Softmax and dropout
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            out = torch.matmul(attn_weights, V)  # [B, H, L, D_H]

            # Reshape and project back
            out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
            out = self.w_o(out)

            if use_cache:
                return out, (K, V)
            return out

        except Exception as e:
            rat_logger.error(f"AdaptivePolicyAttention forward failed: {e}")
            raise
