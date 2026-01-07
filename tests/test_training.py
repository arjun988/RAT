#!/usr/bin/env python3
"""
Unit tests for training components in RAT
"""

import torch
import pytest
from unittest.mock import Mock
from trainer import RATTrainer
from rat_model import RAT


class TestRATTrainer:
    """Test cases for RAT Trainer"""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        return tokenizer

    @pytest.fixture
    def small_model(self):
        """Create small RAT model for testing"""
        return RAT(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=32
        )

    @pytest.fixture
    def trainer(self, small_model, mock_tokenizer):
        """Create trainer for testing"""
        return RATTrainer(
            model=small_model,
            tokenizer=mock_tokenizer,
            lr=1e-3,
            max_steps=5,
            device='cpu',
            accum_steps=1
        )

    def test_initialization(self, trainer):
        """Test trainer initialization"""
        assert trainer.lr == 1e-3
        assert trainer.max_steps == 5
        assert trainer.device == 'cpu'
        assert trainer.accum_steps == 1

    def test_train_step(self, trainer):
        """Test single training step"""
        batch_size, seq_len = 2, 16
        input_seq = torch.randint(1, 1000, (batch_size, seq_len))
        target_seq = torch.randint(1, 1000, (batch_size, seq_len))

        metrics = trainer.train_step(input_seq, target_seq)

        required_keys = ['loss', 'learning_rate', 'step', 'perplexity']
        for key in required_keys:
            assert key in metrics

        assert metrics['loss'] > 0
        assert metrics['perplexity'] > 1

    def test_multiple_train_steps(self, trainer):
        """Test multiple training steps"""
        batch_size, seq_len = 2, 16

        initial_step = trainer.get_training_stats()['current_step']

        # Run a few training steps
        for _ in range(3):
            input_seq = torch.randint(1, 1000, (batch_size, seq_len))
            target_seq = torch.randint(1, 1000, (batch_size, seq_len))
            trainer.train_step(input_seq, target_seq)

        final_step = trainer.get_training_stats()['current_step']
        assert final_step == initial_step + 3

    def test_training_stats(self, trainer):
        """Test training statistics"""
        stats = trainer.get_training_stats()

        required_keys = ['current_step', 'elapsed_time']
        for key in required_keys:
            assert key in stats

        assert isinstance(stats['current_step'], int)
        assert isinstance(stats['elapsed_time'], float)

    def test_optimizer_setup(self, trainer):
        """Test optimizer setup"""
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_gradient_accumulation(self):
        """Test gradient accumulation"""
        model = RAT(vocab_size=1000, d_model=64, n_layers=1, n_heads=2, max_seq_len=16)
        tokenizer = Mock()
        tokenizer.pad_token_id = 0

        trainer = RATTrainer(
            model=model,
            tokenizer=tokenizer,
            lr=1e-3,
            max_steps=5,
            device='cpu',
            accum_steps=2  # Accumulate gradients over 2 steps
        )

        batch_size, seq_len = 2, 8

        # First step - gradients accumulated but not applied
        input_seq1 = torch.randint(1, 1000, (batch_size, seq_len))
        target_seq1 = torch.randint(1, 1000, (batch_size, seq_len))
        metrics1 = trainer.train_step(input_seq1, target_seq1)

        # Second step - gradients applied
        input_seq2 = torch.randint(1, 1000, (batch_size, seq_len))
        target_seq2 = torch.randint(1, 1000, (batch_size, seq_len))
        metrics2 = trainer.train_step(input_seq2, target_seq2)

        assert metrics1['step'] == 1
        assert metrics2['step'] == 2
