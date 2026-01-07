#!/usr/bin/env python3
"""
Unit tests for transformer components in RAT
"""

import torch
import pytest
from rat_model import RATBlock, RAT


class TestRATBlock:
    """Test cases for RAT Block"""

    @pytest.fixture
    def rat_block(self):
        """Create RAT block for testing"""
        d_model = 256
        n_heads = 8
        n_policies = 3
        return RATBlock(d_model, n_heads, n_policies)

    def test_initialization(self, rat_block):
        """Test RAT block initialization"""
        assert rat_block.d_model == 256
        assert rat_block.n_heads == 8
        assert rat_block.n_policies == 3

    def test_forward_pass(self, rat_block):
        """Test basic forward pass"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output, kv_cache = rat_block(x)
        assert output.shape == x.shape
        assert kv_cache is not None

    def test_kv_cache_functionality(self, rat_block):
        """Test KV cache functionality"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        # First pass
        output1, kv_cache1 = rat_block(x, use_cache=True)
        assert output1.shape == x.shape
        assert kv_cache1 is not None

        # Second pass with cache
        x2 = torch.randn(batch_size, 8, d_model)
        output2, kv_cache2 = rat_block(x2, kv_cache=kv_cache1, use_cache=True)
        assert output2.shape == x2.shape

    def test_residual_connections(self, rat_block):
        """Test residual connections"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output, _ = rat_block(x)

        # Output should be different from input (due to residual + layer norm)
        assert not torch.allclose(output, x, atol=1e-6)


class TestRATModel:
    """Test cases for full RAT model"""

    @pytest.fixture
    def rat_model(self):
        """Create RAT model for testing"""
        vocab_size = 10000
        d_model = 256
        n_layers = 4
        n_heads = 8
        max_seq_len = 128
        return RAT(vocab_size, d_model, n_layers, n_heads, max_seq_len)

    def test_initialization(self, rat_model):
        """Test RAT model initialization"""
        assert rat_model.vocab_size == 10000
        assert rat_model.d_model == 256
        assert rat_model.n_layers == 4
        assert rat_model.n_heads == 8
        assert rat_model.max_seq_len == 128

    def test_forward_pass(self, rat_model):
        """Test basic forward pass"""
        batch_size, seq_len = 2, 32
        vocab_size = 10000
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = rat_model(x)
        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_with_kv_cache(self, rat_model):
        """Test forward pass with KV cache"""
        batch_size, seq_len = 2, 16
        vocab_size = 10000
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, kv_cache = rat_model(x, use_cache=True)
        assert logits.shape == (batch_size, seq_len, vocab_size)
        assert len(kv_cache) == rat_model.n_layers

    def test_memory_usage(self, rat_model):
        """Test memory usage reporting"""
        mem_stats = rat_model.get_memory_usage()
        assert 'total_parameters' in mem_stats
        assert 'memory_mb' in mem_stats
        assert mem_stats['total_parameters'] > 0
        assert mem_stats['memory_mb'] > 0

    def test_logits_probabilities(self, rat_model):
        """Test that logits can be converted to probabilities"""
        batch_size, seq_len = 2, 16
        vocab_size = 10000
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits = rat_model(x)
        probs = torch.softmax(logits, dim=-1)

        # Probabilities should sum to 1
        prob_sums = probs.sum(dim=-1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums))

    def test_different_sequence_lengths(self, rat_model):
        """Test model with different sequence lengths"""
        vocab_size = 10000
        batch_size = 2

        # Test various sequence lengths
        for seq_len in [1, 8, 32, 64]:
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits = rat_model(x)
            assert logits.shape == (batch_size, seq_len, vocab_size)
