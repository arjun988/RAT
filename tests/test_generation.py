#!/usr/bin/env python3
"""
Unit tests for generation components in RAT
"""

import torch
import pytest
from unittest.mock import Mock
from generation import RATGenerator
from rat_model import RAT


class TestRATGenerator:
    """Test cases for RAT Generator"""

    @pytest.fixture
    def mock_tokenizer(self):
        """Create mock tokenizer"""
        tokenizer = Mock()
        tokenizer.pad_token_id = 0
        tokenizer.eos_token_id = 1

        def mock_call(text, return_tensors=None):
            # Mock tokenization
            tokens = [2] + [ord(c) % 1000 + 3 for c in text]  # Start from 2, avoid special tokens
            return {"input_ids": torch.tensor([tokens])}

        def mock_decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            if isinstance(token_ids[0], list):
                token_ids = token_ids[0]
            # Mock decoding
            text = ""
            for token in token_ids:
                if token >= 3:
                    char_code = (token - 3) % 256
                    text += chr(char_code)
                elif token == 2:
                    text += "[START]"
            return text

        tokenizer.__call__ = mock_call
        tokenizer.decode = mock_decode
        return tokenizer

    @pytest.fixture
    def small_model(self):
        """Create small RAT model for testing"""
        return RAT(
            vocab_size=1000,
            d_model=128,
            n_layers=2,
            n_heads=4,
            max_seq_len=64
        )

    @pytest.fixture
    def generator(self, small_model, mock_tokenizer):
        """Create generator for testing"""
        return RATGenerator(small_model, mock_tokenizer, device='cpu')

    def test_initialization(self, generator, small_model, mock_tokenizer):
        """Test generator initialization"""
        assert generator.model == small_model
        assert generator.tokenizer == mock_tokenizer
        assert generator.device == 'cpu'

    def test_generate_basic(self, generator):
        """Test basic generation"""
        prompt = "Hello"
        result = generator.generate(prompt, max_len=10)

        assert isinstance(result, str)
        assert len(result) > 0
        assert prompt in result or "[START]" in result

    def test_generate_with_temperature(self, generator):
        """Test generation with temperature"""
        prompt = "Hello"

        # Greedy decoding (temperature=0)
        result_greedy = generator.generate(prompt, max_len=5, temperature=0.0)
        assert isinstance(result_greedy, str)

        # Random sampling (temperature > 0)
        result_random = generator.generate(prompt, max_len=5, temperature=1.0)
        assert isinstance(result_random, str)

    def test_generate_with_top_k(self, generator):
        """Test generation with top-k sampling"""
        prompt = "Hello"

        result = generator.generate(prompt, max_len=5, top_k=10)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_with_top_p(self, generator):
        """Test generation with top-p sampling"""
        prompt = "Hello"

        result = generator.generate(prompt, max_len=5, top_p=0.9)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_greedy_vs_sampling(self, generator):
        """Test greedy vs sampling consistency"""
        prompt = "Hello"

        # Greedy should be deterministic
        result1 = generator.generate(prompt, max_len=5, do_sample=False, temperature=0.0)
        result2 = generator.generate(prompt, max_len=5, do_sample=False, temperature=0.0)
        assert result1 == result2

    def test_max_length_constraint(self, generator):
        """Test max length constraint"""
        prompt = "Hello"

        # Short generation
        result_short = generator.generate(prompt, max_len=3)
        assert len(result_short.split()) <= 5  # Approximate check

        # Longer generation
        result_long = generator.generate(prompt, max_len=10)
        assert len(result_long) >= len(result_short)

    def test_empty_prompt(self, generator):
        """Test generation with empty prompt"""
        result = generator.generate("", max_len=5)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_special_tokens(self, generator):
        """Test handling of special tokens"""
        prompt = "Hello world"

        result = generator.generate(prompt, max_len=8)
        assert isinstance(result, str)

        # Should contain some text
        assert len(result.strip()) > 0
