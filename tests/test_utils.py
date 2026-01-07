#!/usr/bin/env python3
"""
Unit tests for utility functions in RAT
"""

import pytest
from rat_model import validate_model_config


class TestConfigurationValidation:
    """Test cases for configuration validation"""

    def test_valid_config(self):
        """Test valid configuration"""
        valid_config = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'max_seq_len': 1024
        }

        validated = validate_model_config(valid_config)
        assert validated['vocab_size'] == 10000
        assert validated['d_model'] == 512
        assert validated['n_layers'] == 6
        assert validated['n_heads'] == 8
        assert validated['max_seq_len'] == 1024

    def test_invalid_vocab_size(self):
        """Test invalid vocab size"""
        invalid_config = {
            'vocab_size': -1,  # Invalid
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8
        }

        with pytest.raises(ValueError, match="vocab_size must be positive"):
            validate_model_config(invalid_config)

    def test_invalid_d_model(self):
        """Test invalid d_model"""
        invalid_config = {
            'vocab_size': 10000,
            'd_model': 0,  # Invalid
            'n_layers': 6,
            'n_heads': 8
        }

        with pytest.raises(ValueError, match="d_model must be positive"):
            validate_model_config(invalid_config)

    def test_invalid_n_layers(self):
        """Test invalid n_layers"""
        invalid_config = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 0,  # Invalid
            'n_heads': 8
        }

        with pytest.raises(ValueError, match="n_layers must be positive"):
            validate_model_config(invalid_config)

    def test_invalid_n_heads(self):
        """Test invalid n_heads"""
        invalid_config = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': -1  # Invalid
        }

        with pytest.raises(ValueError, match="n_heads must be positive"):
            validate_model_config(invalid_config)

    def test_invalid_max_seq_len(self):
        """Test invalid max_seq_len"""
        invalid_config = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'max_seq_len': 0  # Invalid
        }

        with pytest.raises(ValueError, match="max_seq_len must be positive"):
            validate_model_config(invalid_config)

    def test_d_model_n_heads_compatibility(self):
        """Test d_model and n_heads compatibility"""
        invalid_config = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 7  # 512 not divisible by 7
        }

        with pytest.raises(ValueError, match="d_model must be divisible by n_heads"):
            validate_model_config(invalid_config)

    def test_default_values(self):
        """Test default value assignment"""
        minimal_config = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8
        }

        validated = validate_model_config(minimal_config)
        assert 'max_seq_len' in validated
        assert validated['max_seq_len'] == 1024  # Default value

    def test_extra_fields_preserved(self):
        """Test that extra fields are preserved"""
        config_with_extra = {
            'vocab_size': 10000,
            'd_model': 512,
            'n_layers': 6,
            'n_heads': 8,
            'custom_field': 'custom_value'
        }

        validated = validate_model_config(config_with_extra)
        assert validated['custom_field'] == 'custom_value'
