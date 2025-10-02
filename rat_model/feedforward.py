import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, multiple_of=256, dropout=0.1):
        super().__init__()
        d_ff = d_ff or int(2 * d_model * 4 / 3)
        d_ff = multiple_of * ((d_ff + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
