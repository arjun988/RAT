#!/usr/bin/env python3
"""
Unit tests for dataset components in RAT
"""

import torch
import pytest
from dataset import RATDataset, RATDataLoader


class TestRATDataset:
    """Test cases for RAT Dataset"""

    @pytest.fixture
    def mock_texts(self):
        """Create mock tokenized texts for testing"""
        return [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Sufficient length
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Shorter
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Longer
        ]

    @pytest.fixture
    def dataset(self, mock_texts):
        """Create dataset for testing"""
        seq_len = 8
        return RATDataset(mock_texts, seq_len=seq_len)

    def test_initialization(self, dataset, mock_texts):
        """Test dataset initialization"""
        assert len(dataset) == len(mock_texts)
        assert dataset.seq_len == 8

    def test_getitem(self, dataset):
        """Test dataset item retrieval"""
        input_ids, target_ids = dataset[0]

        assert isinstance(input_ids, torch.Tensor)
        assert isinstance(target_ids, torch.Tensor)
        assert input_ids.shape == (8,)  # seq_len
        assert target_ids.shape == (8,)  # seq_len

        # Target should be shifted by 1 compared to input
        assert torch.equal(input_ids[1:], target_ids[:-1])

    def test_sequence_padding(self, dataset):
        """Test sequence padding for shorter sequences"""
        # Get item from shorter sequence (index 1)
        input_ids, target_ids = dataset[1]

        assert input_ids.shape == (8,)
        assert target_ids.shape == (8,)

        # Check that padding works correctly
        assert input_ids[-1] == 0  # Assuming 0 is pad token
        assert target_ids[-1] == 0

    def test_statistics(self, dataset, mock_texts):
        """Test dataset statistics"""
        stats = dataset.get_statistics()

        assert 'num_sequences' in stats
        assert 'avg_length' in stats
        assert stats['num_sequences'] == len(mock_texts)

        # Calculate expected average length
        lengths = [len(text) for text in mock_texts]
        expected_avg = sum(lengths) / len(lengths)
        assert abs(stats['avg_length'] - expected_avg) < 0.1

    def test_edge_cases(self):
        """Test edge cases"""
        # Empty dataset
        empty_dataset = RATDataset([], seq_len=8)
        assert len(empty_dataset) == 0

        # Single token sequences
        single_tokens = [[1], [2], [3]]
        single_dataset = RATDataset(single_tokens, seq_len=4)
        assert len(single_dataset) == 3

        input_ids, target_ids = single_dataset[0]
        assert input_ids.shape == (4,)
        assert target_ids.shape == (4,)


class TestRATDataLoader:
    """Test cases for RAT DataLoader"""

    @pytest.fixture
    def mock_texts(self):
        """Create mock tokenized texts"""
        return [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,  # Long sequence
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 3,  # Medium sequence
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 7,  # Very long sequence
        ]

    @pytest.fixture
    def dataloader(self, mock_texts):
        """Create dataloader for testing"""
        dataset = RATDataset(mock_texts, seq_len=8)
        return RATDataLoader(dataset, batch_size=2, shuffle=False)

    def test_initialization(self, dataloader):
        """Test dataloader initialization"""
        assert dataloader.batch_size == 2
        assert not dataloader.shuffle

    def test_iteration(self, dataloader):
        """Test dataloader iteration"""
        batches = list(dataloader)

        # Should have at least one batch
        assert len(batches) > 0

        for batch_input, batch_target in batches:
            assert isinstance(batch_input, torch.Tensor)
            assert isinstance(batch_target, torch.Tensor)
            assert batch_input.shape[0] <= 2  # batch_size
            assert batch_target.shape[0] <= 2  # batch_size
            assert batch_input.shape[1] == 8  # seq_len
            assert batch_target.shape[1] == 8  # seq_len

    def test_batch_shapes(self, dataloader):
        """Test batch shapes"""
        for batch_input, batch_target in dataloader:
            # Check batch dimensions
            assert batch_input.dim() == 2  # (batch_size, seq_len)
            assert batch_target.dim() == 2  # (batch_size, seq_len)

            # Check sequence length
            assert batch_input.shape[1] == 8
            assert batch_target.shape[1] == 8
            break  # Just test first batch

    def test_empty_dataloader(self):
        """Test empty dataloader"""
        empty_dataset = RATDataset([], seq_len=8)
        empty_dataloader = RATDataLoader(empty_dataset, batch_size=2)

        batches = list(empty_dataloader)
        assert len(batches) == 0
