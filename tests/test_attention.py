#!/usr/bin/env python3
"""
Unit tests for attention mechanisms in RAT
"""

import torch
import pytest
from rat_model import AdaptivePolicyAttention, RotaryPositionEmbedding


class TestAdaptivePolicyAttention:
    """Test cases for Adaptive Policy Attention"""

    @pytest.fixture
    def attention(self):
        """Create attention module for testing"""
        d_model = 256
        n_heads = 8
        n_policies = 3
        return AdaptivePolicyAttention(d_model, n_heads, n_policies)

    @pytest.fixture
    def rope(self):
        """Create RoPE module for testing"""
        dim = 64
        max_seq_len = 128
        return RotaryPositionEmbedding(dim, max_seq_len)

    def test_initialization(self, attention):
        """Test attention module initialization"""
        assert attention.d_model == 256
        assert attention.n_heads == 8
        assert attention.n_policies == 3
        assert attention.head_dim == 32

    def test_forward_pass(self, attention):
        """Test basic forward pass"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)
        assert output.shape == x.shape

    def test_with_rope(self, attention, rope):
        """Test attention with RoPE"""
        attention.rope = rope
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output = attention(x)
        assert output.shape == x.shape

    def test_kv_cache(self, attention):
        """Test KV cache functionality"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        # First pass - create cache
        output1, kv_cache = attention(x, use_cache=True)
        assert output1.shape == x.shape
        assert len(kv_cache) == 2  # K and V caches

        # Second pass - use cache
        x2 = torch.randn(batch_size, 8, d_model)
        output2, kv_cache2 = attention(x2, kv_cache=kv_cache, use_cache=True)
        assert output2.shape == x2.shape

    def test_policy_selection(self, attention):
        """Test policy selection mechanism"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        # Test different policy selection
        attention.policy_selector = torch.tensor([0.0, 1.0, 0.0])  # Select policy 1
        output = attention(x)
        assert output.shape == x.shape

    def test_attention_masking(self, attention):
        """Test attention masking"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        attention.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

        output = attention(x)
        assert output.shape == x.shape


class TestRotaryPositionEmbedding:
    """Test cases for Rotary Position Embedding"""

    def test_initialization(self):
        """Test RoPE initialization"""
        dim = 64
        max_seq_len = 128
        rope = RotaryPositionEmbedding(dim, max_seq_len)

        assert rope.dim == dim
        assert rope.max_seq_len == max_seq_len

    def test_forward_pass(self):
        """Test RoPE forward pass"""
        dim = 64
        max_seq_len = 128
        rope = RotaryPositionEmbedding(dim, max_seq_len)

        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, dim)

        cos, sin = rope(x, seq_len=seq_len)
        assert cos.shape == (batch_size, 1, seq_len, dim)
        assert sin.shape == (batch_size, 1, seq_len, dim)

    def test_different_sequence_lengths(self):
        """Test RoPE with different sequence lengths"""
        dim = 64
        max_seq_len = 128
        rope = RotaryPositionEmbedding(dim, max_seq_len)

        # Test shorter sequence
        x_short = torch.randn(2, 16, dim)
        cos_short, sin_short = rope(x_short, seq_len=16)
        assert cos_short.shape == (2, 1, 16, dim)

        # Test longer sequence
        x_long = torch.randn(2, 64, dim)
        cos_long, sin_long = rope(x_long, seq_len=64)
        assert cos_long.shape == (2, 1, 64, dim)
