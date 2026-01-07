#!/usr/bin/env python3
"""
Unit tests for feedforward networks in RAT
"""

import torch
import pytest
from rat_model import SwiGLUFeedForward


class TestSwiGLUFeedForward:
    """Test cases for SwiGLU Feed Forward"""

    @pytest.fixture
    def feedforward(self):
        """Create feedforward module for testing"""
        d_model = 256
        d_ff = 1024
        return SwiGLUFeedForward(d_model, d_ff)

    def test_initialization(self, feedforward):
        """Test feedforward module initialization"""
        assert feedforward.d_model == 256
        assert feedforward.d_ff == 1024

    def test_forward_pass(self, feedforward):
        """Test basic forward pass"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        output = feedforward(x)
        assert output.shape == x.shape

    def test_swiglu_activation(self, feedforward):
        """Test SwiGLU activation function"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model)

        # Test that output is different from regular FF
        output = feedforward(x)

        # SwiGLU should produce non-zero outputs
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_gradient_flow(self, feedforward):
        """Test gradient flow through the network"""
        batch_size, seq_len, d_model = 2, 16, 256
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)

        output = feedforward(x)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_different_input_sizes(self, feedforward):
        """Test feedforward with different input sizes"""
        # Test various batch sizes and sequence lengths
        test_cases = [
            (1, 1, 256),   # Single token
            (4, 32, 256),  # Standard batch
            (8, 128, 256), # Large batch
        ]

        for batch_size, seq_len, d_model in test_cases:
            x = torch.randn(batch_size, seq_len, d_model)
            output = feedforward(x)
            assert output.shape == x.shape
