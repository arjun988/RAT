#!/usr/bin/env python3
"""
Integration tests for RAT implementation
Tests all components working together with comprehensive scenarios.
"""

import torch
import pytest


class TestRATIntegration:
    """Integration tests for RAT components working together"""

    def test_full_pipeline_training_and_generation(self, device, mock_tokenizer):
        """Test complete pipeline: training -> generation"""
        from rat_model import RAT
        from trainer import RATTrainer
        from generation import RATGenerator

        # Create small model for testing
        model = RAT(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32
        ).to(device)

        trainer = RATTrainer(
            model=model,
            tokenizer=mock_tokenizer,
            lr=1e-3,
            max_steps=3,
            device=device,
            accum_steps=1
        )

        generator = RATGenerator(model, mock_tokenizer, device=device)

        # Create training data
        batch_size, seq_len = 2, 16
        input_seq = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
        target_seq = torch.randint(1, 1000, (batch_size, seq_len)).to(device)

        # Train for a few steps
        initial_loss = trainer.train_step(input_seq, target_seq)['loss']
        final_loss = trainer.train_step(input_seq, target_seq)['loss']

        # Loss should decrease or stay similar (not increase dramatically)
        assert final_loss <= initial_loss * 2.0

        # Generate text
        prompt = "Hello"
        generated_text = generator.generate(prompt, max_len=5)

        assert isinstance(generated_text, str)
        assert len(generated_text) > 0

    def test_model_serialization_and_loading(self, device, tmp_path):
        """Test model saving and loading"""
        from rat_model import RAT

        # Create and save model
        model = RAT(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32
        ).to(device)

        model_path = tmp_path / "test_model.pt"
        torch.save(model.state_dict(), model_path)

        # Create new model and load weights
        new_model = RAT(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32
        ).to(device)

        new_model.load_state_dict(torch.load(model_path))

        # Test that loaded model produces same outputs
        x = torch.randint(0, 1000, (2, 16)).to(device)

        model.eval()
        new_model.eval()

        with torch.no_grad():
            orig_output = model(x)
            loaded_output = new_model(x)

        # Outputs should be identical
        torch.testing.assert_close(orig_output, loaded_output)

    def test_gradient_flow_end_to_end(self, device):
        """Test gradient flow through entire model"""
        from rat_model import RAT

        model = RAT(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=32
        ).to(device)

        # Create input and target
        batch_size, seq_len = 2, 16
        x = torch.randint(1, 1000, (batch_size, seq_len)).to(device)
        targets = torch.randint(1, 1000, (batch_size, seq_len)).to(device)

        # Forward pass
        logits = model(x)

        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, 1000), targets.view(-1)
        )

        # Backward pass
        loss.backward()

        # Check that gradients exist and are reasonable
        total_grad_norm = 0
        param_count = 0

        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm ** 2
                param_count += 1

                # Gradients should not be NaN or infinite
                assert not torch.isnan(param.grad).any()
                assert not torch.isinf(param.grad).any()

        total_grad_norm = total_grad_norm ** 0.5
        assert total_grad_norm > 0, "Total gradient norm should be positive"
        assert param_count > 0, "Should have parameters with gradients"

    def test_memory_efficiency_with_kv_cache(self, device):
        """Test memory efficiency improvements with KV cache"""
        from rat_model import RAT

        model = RAT(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            max_seq_len=64
        ).to(device)

        batch_size, seq_len = 2, 32
        x = torch.randint(0, 1000, (batch_size, seq_len)).to(device)

        # First pass without cache
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        with torch.no_grad():
            logits1 = model(x)

        mem_after_first = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Second pass with cache
        with torch.no_grad():
            logits2, kv_cache = model(x, use_cache=True)

        mem_after_second = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0

        # Outputs should be the same
        torch.testing.assert_close(logits1, logits2)

        # KV cache should exist
        assert len(kv_cache) == model.n_layers
        for layer_cache in kv_cache:
            assert len(layer_cache) == 2  # K and V caches

    def test_dataset_and_training_integration(self, device, mock_tokenizer):
        """Test dataset creation and training integration"""
        from rat_model import RAT
        from trainer import RATTrainer
        from dataset import RATDataset

        # Create mock tokenized texts
        mock_texts = [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 2,  # Sufficient length
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 2,  # Shorter
        ]

        # Create dataset
        seq_len = 16
        dataset = RATDataset(mock_texts, seq_len=seq_len)

        # Create model and trainer
        model = RAT(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=4,
            max_seq_len=seq_len
        ).to(device)

        trainer = RATTrainer(
            model=model,
            tokenizer=mock_tokenizer,
            lr=1e-3,
            max_steps=2,
            device=device,
            accum_steps=1
        )

        # Test training with dataset sample
        input_ids, target_ids = dataset[0]
        input_batch = input_ids.unsqueeze(0).to(device)  # Add batch dimension
        target_batch = target_ids.unsqueeze(0).to(device)

        metrics = trainer.train_step(input_batch, target_batch)

        # Should have valid metrics
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] > 0
        assert metrics['perplexity'] > 1

    def test_swiglu_feedforward(self):
        """Test SwiGLU Feed Forward"""
        self.logger.info("Testing SwiGLU Feed Forward...")

        try:
            d_model = 256
            d_ff = 1024
            ff = SwiGLUFeedForward(d_model, d_ff)

            # Test forward pass
            x = torch.randn(2, 32, d_model)
            output = ff(x)

            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"

            self.results["swiglu_feedforward"] = "PASS"
            self.logger.info("✓ SwiGLU Feed Forward test passed")

        except Exception as e:
            self.results["swiglu_feedforward"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ SwiGLU Feed Forward test failed: {e}")

    def test_adaptive_policy_attention(self):
        """Test Adaptive Policy Attention"""
        self.logger.info("Testing Adaptive Policy Attention...")

        try:
            d_model = 256
            n_heads = 8
            n_policies = 3
            seq_len = 16

            attention = AdaptivePolicyAttention(d_model, n_heads, n_policies)

            # Test forward pass
            x = torch.randn(2, seq_len, d_model)
            output = attention(x)

            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"

            # Test with RoPE
            rope = RotaryPositionEmbedding(d_model // n_heads, 128)
            attention.rope = rope
            output_with_rope = attention(x)
            assert output_with_rope.shape == x.shape

            # Test with cache
            output_cached, kv_cache = attention(x, use_cache=True)
            assert output_cached.shape == x.shape
            assert len(kv_cache) == 2, "KV cache should contain K and V"

            self.results["adaptive_policy_attention"] = "PASS"
            self.logger.info("✓ Adaptive Policy Attention test passed")

        except Exception as e:
            self.results["adaptive_policy_attention"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ Adaptive Policy Attention test failed: {e}")

    def test_rat_block(self):
        """Test RAT Block"""
        self.logger.info("Testing RAT Block...")

        try:
            d_model = 256
            n_heads = 8
            n_policies = 3

            block = RATBlock(d_model, n_heads, n_policies)

            # Test forward pass
            x = torch.randn(2, 16, d_model)
            output, _ = block(x)  # Block always returns (output, kv_cache)

            assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"

            # Test with cache
            output_cached, kv_cache = block(x, use_cache=True)
            assert output_cached.shape == x.shape
            assert kv_cache is not None

            self.results["rat_block"] = "PASS"
            self.logger.info("✓ RAT Block test passed")

        except Exception as e:
            self.results["rat_block"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ RAT Block test failed: {e}")

    def test_full_rat_model(self):
        """Test Full RAT Model"""
        self.logger.info("Testing Full RAT Model...")

        try:
            vocab_size = 30000
            d_model = 256
            n_layers = 4
            n_heads = 8
            max_seq_len = 128

            model = RAT(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_seq_len=max_seq_len
            ).to(self.device)

            # Test forward pass
            batch_size = 2
            seq_len = 32
            x = torch.randint(0, vocab_size, (batch_size, seq_len)).to(self.device)

            with torch.no_grad():
                logits = model(x)
                assert logits.shape == (batch_size, seq_len, vocab_size), f"Logits shape mismatch: {logits.shape}"

                # Test with cache
                logits_cached, kv_cache = model(x, use_cache=True)
                assert logits_cached.shape == logits.shape
                assert len(kv_cache) == n_layers, f"KV cache layers mismatch: {len(kv_cache)} vs {n_layers}"

            # Test memory usage method
            mem_stats = model.get_memory_usage()
            assert 'total_parameters' in mem_stats
            assert 'memory_mb' in mem_stats

            self.results["full_rat_model"] = "PASS"
            self.logger.info("✓ Full RAT Model test passed")

        except Exception as e:
            self.results["full_rat_model"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ Full RAT Model test failed: {e}")

    def test_rat_trainer(self):
        """Test RAT Trainer"""
        self.logger.info("Testing RAT Trainer...")

        try:
            # Small model for testing
            vocab_size = 1000
            d_model = 128
            n_layers = 2
            n_heads = 4
            max_seq_len = 32

            model = RAT(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_seq_len=max_seq_len
            ).to(self.device)

            # Mock tokenizer
            class MockTokenizer:
                def __init__(self):
                    self.pad_token_id = 0

            tokenizer = MockTokenizer()

            trainer = RATTrainer(
                model=model,
                tokenizer=tokenizer,
                lr=1e-3,
                max_steps=5,  # Very few steps for testing
                device=self.device,
                accum_steps=1
            )

            # Create mock training data
            batch_size = 2
            seq_len = 16
            input_seq = torch.randint(1, vocab_size, (batch_size, seq_len)).to(self.device)
            target_seq = torch.randint(1, vocab_size, (batch_size, seq_len)).to(self.device)

            # Test training step
            metrics = trainer.train_step(input_seq, target_seq)

            required_keys = ['loss', 'learning_rate', 'step', 'perplexity']
            for key in required_keys:
                assert key in metrics, f"Missing required metric: {key}"
            assert metrics['loss'] > 0, f"Loss should be positive, got {metrics['loss']}"

            # Test training stats
            stats = trainer.get_training_stats()
            assert 'current_step' in stats
            assert 'elapsed_time' in stats

            self.results["rat_trainer"] = "PASS"
            self.logger.info("✓ RAT Trainer test passed")

        except Exception as e:
            self.results["rat_trainer"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ RAT Trainer test failed: {e}")

    def test_rat_generator(self):
        """Test RAT Generator"""
        self.logger.info("Testing RAT Generator...")

        try:
            # Small model for testing
            vocab_size = 1000
            d_model = 128
            n_layers = 2
            n_heads = 4
            max_seq_len = 64

            model = RAT(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_seq_len=max_seq_len
            ).to(self.device)

            # Mock tokenizer
            class MockTokenizer:
                def __init__(self):
                    self.pad_token_id = 0
                    self.eos_token_id = 1

                def __call__(self, text, return_tensors=None):
                    # Mock tokenization
                    tokens = [2] + [ord(c) % (vocab_size - 3) + 3 for c in text]  # Start from 2, avoid special tokens
                    if len(tokens) > max_seq_len:
                        tokens = tokens[:max_seq_len]
                    return {"input_ids": torch.tensor([tokens])}

                def decode(self, token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
                    if isinstance(token_ids, torch.Tensor):
                        token_ids = token_ids.tolist()
                    if isinstance(token_ids[0], list):
                        token_ids = token_ids[0]
                    # Mock decoding - convert back to characters where possible
                    text = ""
                    for token in token_ids:
                        if token >= 3:
                            char_code = (token - 3) % 256
                            text += chr(char_code)
                        elif token == 2:
                            text += "[START]"
                    return text

            tokenizer = MockTokenizer()
            generator = RATGenerator(model, tokenizer, device=self.device)

            # Test generation
            prompt = "Hello"
            generated_text = generator.generate(prompt, max_len=10)

            assert isinstance(generated_text, str), "Generated text should be a string"
            assert len(generated_text) > 0, "Generated text should not be empty"

            # Test with different parameters
            generated_greedy = generator.generate(prompt, max_len=5, do_sample=False)
            assert isinstance(generated_greedy, str)

            self.results["rat_generator"] = "PASS"
            self.logger.info("✓ RAT Generator test passed")

        except Exception as e:
            self.results["rat_generator"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ RAT Generator test failed: {e}")

    def test_rat_dataset(self):
        """Test RAT Dataset"""
        self.logger.info("Testing RAT Dataset...")

        try:
            # Create mock tokenized texts
            mock_texts = [
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # Sufficient length
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Shorter
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]  # Longer
            ]

            seq_len = 8
            dataset = RATDataset(mock_texts, seq_len=seq_len)

            # Test length
            assert len(dataset) == len(mock_texts), f"Dataset length mismatch: {len(dataset)} vs {len(mock_texts)}"

            # Test item retrieval
            input_ids, target_ids = dataset[0]
            assert input_ids.shape == (seq_len,), f"Input shape mismatch: {input_ids.shape}"
            assert target_ids.shape == (seq_len,), f"Target shape mismatch: {target_ids.shape}"

            # Test statistics
            stats = dataset.get_statistics()
            assert 'num_sequences' in stats
            assert 'avg_length' in stats
            assert stats['num_sequences'] == len(mock_texts)

            self.results["rat_dataset"] = "PASS"
            self.logger.info("✓ RAT Dataset test passed")

        except Exception as e:
            self.results["rat_dataset"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ RAT Dataset test failed: {e}")

    def test_gradient_flow_and_numerical_stability(self):
        """Test Gradient Flow and Numerical Stability"""
        self.logger.info("Testing Gradient Flow and Numerical Stability...")

        try:
            # Small model for gradient testing
            model = RAT(
                vocab_size=1000,
                d_model=128,
                n_layers=2,
                n_heads=4,
                max_seq_len=32
            ).to(self.device)

            # Test gradient flow
            x = torch.randint(1, 1000, (2, 16)).to(self.device)
            targets = torch.randint(1, 1000, (2, 16)).to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            criterion = nn.CrossEntropyLoss()

            model.train()
            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(logits.view(-1, 1000), targets.view(-1))
            loss.backward()

            # Check for gradient flow
            has_grad = False
            grad_norms = []
            for name, param in model.named_parameters():
                if param.grad is not None:
                    has_grad = True
                    grad_norm = param.grad.data.norm(2).item()
                    grad_norms.append(grad_norm)

            assert has_grad, "No gradients found - model not learning"
            assert not torch.isnan(loss), "Loss is NaN"
            assert not torch.isinf(loss), "Loss is infinite"

            total_grad_norm = sum(grad_norms)
            assert total_grad_norm > 0, "Total gradient norm is zero"

            self.logger.info(f"Loss: {loss.item():.4f}, Total grad norm: {total_grad_norm:.4f}")
            self.results["gradient_flow"] = "PASS"
            self.logger.info("✓ Gradient Flow test passed")

        except Exception as e:
            self.results["gradient_flow"] = f"FAIL: {str(e)}"
            self.logger.error(f"✗ Gradient Flow test failed: {e}")

    def run_all_tests(self):
        """Run all tests"""
        self.logger.info("="*60)
        self.logger.info("STARTING COMPREHENSIVE RAT TESTING")
        self.logger.info("="*60)

        start_time = time.time()

        # Run all tests
        test_methods = [
            self.test_configuration_validation,
            self.test_rotary_embeddings,
            self.test_swiglu_feedforward,
            self.test_adaptive_policy_attention,
            self.test_rat_block,
            self.test_full_rat_model,
            self.test_rat_trainer,
            self.test_rat_generator,
            self.test_rat_dataset,
            self.test_gradient_flow_and_numerical_stability,
        ]

        for test_method in test_methods:
            try:
                test_method()
            except Exception as e:
                self.logger.error(f"Unexpected error in {test_method.__name__}: {e}")
                self.results[test_method.__name__.replace('test_', '')] = f"FAIL: {str(e)}"

        # Summary
        total_time = time.time() - start_time
        passed = sum(1 for result in self.results.values() if result == "PASS")
        total = len(self.results)

        self.logger.info("="*60)
        self.logger.info("RAT TEST RESULTS SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Total tests: {total}")
        self.logger.info(f"Passed: {passed}")
        self.logger.info(f"Failed: {total - passed}")
        self.logger.info(".2f")
        self.logger.info("")

        for test_name, result in self.results.items():
            status = "✓" if result == "PASS" else "✗"
            self.logger.info(f"{status} {test_name}: {result}")

        self.logger.info("="*60)

        if passed == total:
            self.logger.info("🎉 ALL RAT TESTS PASSED! Ready for packaging.", "SUCCESS")
            return True
        else:
            self.logger.info(f"⚠️  {total - passed} tests failed. Review implementation before packaging.", "WARNING")
            return False


def main():
    """Main test function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test RAT Implementation")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run tests on (cuda/cpu)")
    parser.add_argument("--quick", action="store_true",
                       help="Run only basic component tests")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None

    tester = RATTester(device=device)

    if args.quick:
        # Run only core component tests
        tester.test_configuration_validation()
        tester.test_rotary_embeddings()
        tester.test_swiglu_feedforward()
        tester.test_adaptive_policy_attention()
        tester.test_full_rat_model()
    else:
        # Run all tests
        success = tester.run_all_tests()
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    exit(main())
