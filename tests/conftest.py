#!/usr/bin/env python3
"""
Common pytest fixtures and configuration for RAT tests
"""

import torch
import pytest
from unittest.mock import Mock


@pytest.fixture(scope="session")
def device():
    """Get available device for testing"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def mock_tokenizer():
    """Create mock tokenizer for testing"""
    tokenizer = Mock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    tokenizer.bos_token_id = 2

    def mock_call(text, return_tensors=None):
        # Mock tokenization - simple character to token mapping
        tokens = [2] + [ord(c) % 1000 + 3 for c in text]  # Start from 2, avoid special tokens
        if len(tokens) > 512:  # Max length
            tokens = tokens[:512]
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
def small_rat_model():
    """Create small RAT model for testing"""
    from rat_model import RAT
    return RAT(
        vocab_size=1000,
        d_model=64,
        n_layers=2,
        n_heads=4,
        max_seq_len=32
    )


@pytest.fixture
def tiny_rat_model():
    """Create tiny RAT model for fast testing"""
    from rat_model import RAT
    return RAT(
        vocab_size=100,
        d_model=32,
        n_layers=1,
        n_heads=2,
        max_seq_len=16
    )
