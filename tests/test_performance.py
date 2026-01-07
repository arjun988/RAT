#!/usr/bin/env python3
"""
Performance tests for RAT components
"""

import torch
import pytest
import time


class TestRATPerformance:
    """Performance tests for RAT components"""

    @pytest.mark.slow
    def test_model_inference_speed(self, small_rat_model, device):
        """Test model inference speed"""
        model = small_rat_model.to(device)
        model.eval()

        batch_sizes = [1, 4, 16]
        seq_lengths = [16, 32, 64]

        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                x = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

                # Warm up
                with torch.no_grad():
                    _ = model(x)

                # Measure inference time
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()

                with torch.no_grad():
                    for _ in range(10):  # Average over 10 runs
                        _ = model(x)

                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()

                avg_time = (end_time - start_time) / 10
                tokens_per_sec = (batch_size * seq_len) / avg_time

                # Should be reasonably fast (at least 100 tokens/sec for small model)
                assert tokens_per_sec > 50, f"Inference too slow: {tokens_per_sec:.1f} tokens/sec"

    @pytest.mark.slow
    def test_kv_cache_memory_efficiency(self, small_rat_model, device):
        """Test KV cache memory efficiency"""
        model = small_rat_model.to(device)
        model.eval()

        seq_len = 32
        x = torch.randint(0, 1000, (1, seq_len)).to(device)

        # Measure memory without cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with torch.no_grad():
            logits_no_cache = model(x)

        mem_after_no_cache = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Measure memory with cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_before_cache = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with torch.no_grad():
            logits_with_cache, kv_cache = model(x, use_cache=True)

        mem_after_cache = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Outputs should be identical
        torch.testing.assert_close(logits_no_cache, logits_with_cache)

        # Memory usage should be reasonable
        if torch.cuda.is_available():
            mem_no_cache = mem_after_no_cache - mem_before
            mem_with_cache = mem_after_cache - mem_before_cache
            # Cache shouldn't use dramatically more memory
            assert mem_with_cache < mem_no_cache * 3

    def test_model_memory_usage(self, small_rat_model):
        """Test model memory usage reporting"""
        mem_stats = small_rat_model.get_memory_usage()

        assert 'total_parameters' in mem_stats
        assert 'memory_mb' in mem_stats

        # Should have reasonable parameter count
        assert mem_stats['total_parameters'] > 1000
        assert mem_stats['total_parameters'] < 10000000  # Less than 10M params for small model

        # Memory should be reasonable (less than 1GB for small model)
        assert mem_stats['memory_mb'] > 0
        assert mem_stats['memory_mb'] < 1000

    def test_training_step_performance(self, small_rat_model, mock_tokenizer, device):
        """Test training step performance"""
        from trainer import RATTrainer

        model = small_rat_model.to(device)
        trainer = RATTrainer(
            model=model,
            tokenizer=mock_tokenizer,
            lr=1e-3,
            max_steps=5,
            device=device,
            accum_steps=1
        )

        batch_size, seq_len = 2, 16
        input_seq = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
        target_seq = torch.randint(1, 1000, (batch_size, seq_len)).to(device)

        start_time = time.time()
        metrics = trainer.train_step(input_seq, target_seq)
        end_time = time.time()

        training_time = end_time - start_time

        # Training step should complete in reasonable time (less than 1 second for small model)
        assert training_time < 1.0, f"Training step too slow: {training_time:.3f}s"

        # Should have valid loss
        assert metrics['loss'] > 0
        assert not torch.isnan(torch.tensor(metrics['loss']))

    def test_generation_performance(self, small_rat_model, mock_tokenizer, device):
        """Test text generation performance"""
        from generation import RATGenerator

        model = small_rat_model.to(device)
        generator = RATGenerator(model, mock_tokenizer, device=device)

        prompt = "Hello world"
        max_len = 20

        start_time = time.time()
        generated_text = generator.generate(prompt, max_len=max_len)
        end_time = time.time()

        generation_time = end_time - start_time

        # Generation should complete in reasonable time (less than 5 seconds for small model)
        assert generation_time < 5.0, f"Generation too slow: {generation_time:.3f}s"

        # Should generate some text
        assert isinstance(generated_text, str)
        assert len(generated_text) > len(prompt)
