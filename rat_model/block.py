import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from .attention import MultiPolicyRLAttention
from .feedforward import SwiGLUFeedForward

class EnhancedRATBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_policies=3, dropout=0.1, use_rope=True, mlp_ratio=4, use_checkpointing=False):
        super().__init__()
        self.use_checkpointing = use_checkpointing
        self.d_model = d_model
        self.attn_norm = nn.LayerNorm(d_model)
        self.attention = MultiPolicyRLAttention(d_model, n_heads, n_policies=n_policies, dropout=dropout, use_rope=use_rope)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, d_ff=d_model * mlp_ratio, dropout=dropout)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.tconv_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        if x.dim() == 2:
            x = x.unsqueeze(-1).expand(-1, -1, self.d_model)
        def attn_block(x_inner):
            residual = x_inner
            x_inner = self.attn_norm(x_inner)
            if use_cache:
                attn_out, present_kv = self.attention(x_inner, mask, past_kv, use_cache)
            else:
                attn_out = self.attention(x_inner, mask, past_kv, use_cache)
                present_kv = None
            x_inner = residual + self.dropout(attn_out)
            return x_inner, present_kv
        if self.use_checkpointing and self.training and not use_cache:
            x, present_kv = checkpoint.checkpoint(attn_block, x, use_reentrant=False)
        else:
            x, present_kv = attn_block(x)
        residual = x
        x_temp = self.tconv_norm(x)
        x_temp = self.temporal_conv(x_temp.transpose(1, 2)).transpose(1, 2)
        x = residual + self.dropout(x_temp)
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.dropout(self.ffn(x))
        if use_cache:
            return x, present_kv
        return x
