import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .rotary_embedding import apply_rotary_pos_emb

class MultiPolicyRLAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_policies=3, dropout=0.1, use_rope=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_policies = n_policies
        self.use_rope = use_rope
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.policy_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, n_heads),
                nn.Softmax(dim=-1)
            ) for _ in range(n_policies)
        ])
        self.policy_selector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_policies),
            nn.Softmax(dim=-1)
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, mask=None, past_kv=None, use_cache=False):
        B, L, D = x.shape
        Q = self.w_q(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        if hasattr(self, 'rope') and self.use_rope:
            Q, K = apply_rotary_pos_emb(Q, K, *self.rope(x, L))
        if past_kv is not None:
            past_k, past_v = past_kv
            K = torch.cat([past_k, K], dim=2)
            V = torch.cat([past_v, V], dim=2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        seq_len = scores.size(-1)
        if mask is None:
            causal_mask = torch.triu(torch.ones(L, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask[None, None, :, :], -1e9)
        policy_weights = self.policy_selector(x.mean(dim=1))
        head_gates = torch.zeros(B, self.n_heads, 1, 1, device=x.device)
        for i, policy_net in enumerate(self.policy_nets):
            policy_gate = policy_net(x.mean(dim=1))
            head_gates += policy_weights[:, i:i+1, None, None] * policy_gate.view(B, self.n_heads, 1, 1)
        scores = scores * head_gates
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        out = self.w_o(out)
        if use_cache:
            return out, (K, V)
        return out
