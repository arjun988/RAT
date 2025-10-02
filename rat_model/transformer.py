import torch
import torch.nn as nn
from .block import EnhancedRATBlock
from .rotary_embedding import RotaryPositionEmbedding

class EnhancedGPTRAT(nn.Module):
    def __init__(self, vocab_size, d_model=768, n_layers=12, n_heads=12, n_policies=3, max_seq_len=2048, dropout=0.1, mlp_ratio=4, use_rope=True, tie_weights=True, use_checkpointing=False):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        if use_rope:
            self.rope = RotaryPositionEmbedding(d_model // n_heads, max_seq_len)
        self.blocks = nn.ModuleList([
            EnhancedRATBlock(d_model=d_model, n_heads=n_heads, n_policies=n_policies, dropout=dropout, use_rope=use_rope, mlp_ratio=mlp_ratio, use_checkpointing=use_checkpointing)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.token_emb.weight = self.output.weight
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    def forward(self, x, mask=None, past_kvs=None, use_cache=False):
        B, L = x.shape
        token_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :L, :]
        x = self.dropout(token_emb + pos_emb)
        if past_kvs is None:
            past_kvs = [None] * len(self.blocks)
        present_kvs = []
        for i, block in enumerate(self.blocks):
            if hasattr(self, 'rope'):
                block.attention.rope = self.rope
            x, present_kv = block(x, mask=mask, past_kv=past_kvs[i], use_cache=use_cache)
            present_kvs.append(present_kv)
        x = self.norm(x)
        logits = self.output(x)
        if use_cache:
            return logits, present_kvs
        return logits
