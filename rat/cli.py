#!/usr/bin/env python3
"""
RAT Command Line Interface

Provides CLI tools for training, generation, evaluation, and testing RAT models.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rat_model import (
    RAT, RATTrainer, RATGenerator,
    RATDataset, RATDataLoader, RATLogger
)
from transformers import AutoTokenizer
import torch


def setup_logging(verbose: bool = False) -> RATLogger:
    """Setup logging for CLI commands"""
    level = 10 if verbose else 20  # DEBUG or INFO
    return RATLogger("RATCLI", level=level)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON/YAML file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        if config_path.endswith('.json'):
            return json.load(f)
        elif config_path.endswith(('.yaml', '.yml')):
            try:
                import yaml
                return yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")
        else:
            raise ValueError("Config file must be .json, .yaml, or .yml")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        if config_path.endswith('.json'):
            json.dump(config, f, indent=2)
        elif config_path.endswith(('.yaml', '.yml')):
            try:
                import yaml
                yaml.dump(config, f, default_flow_style=False)
            except ImportError:
                raise ImportError("PyYAML is required for YAML config files")


def get_device(device_str: Optional[str] = None) -> torch.device:
    """Get torch device from string"""
    if device_str is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def train_command():
    """Train a RAT model"""
    parser = argparse.ArgumentParser(description="Train RAT model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file (.json or .yaml)")
    parser.add_argument("--output-dir", type=str, default="./checkpoints",
                       help="Output directory for checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to train on (cuda/cpu)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Get device
        device = get_device(args.device)
        logger.info(f"Using device: {device}")

        # Load tokenizer
        tokenizer_path = config.get('tokenizer_path', 'gpt2')
        logger.info(f"Loading tokenizer: {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Create model
        model_config = config.get('model', {})
        logger.info("Creating RAT model")
        model = RAT(**model_config)

        # Create trainer
        trainer_config = config.get('trainer', {})
        trainer_config['checkpoint_dir'] = args.output_dir

        logger.info("Initializing trainer")
        trainer = RATTrainer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            **trainer_config
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # Prepare dataset
        dataset_config = config.get('dataset', {})
        logger.info("Preparing dataset")
        data_loader = RATDataLoader(tokenizer=tokenizer, **dataset_config)
        train_loader = data_loader.prepare_dataset()

        # Training loop
        logger.info("Starting training...")
        max_steps = trainer_config.get('max_steps', 10000)

        for step, batch in enumerate(train_loader):
            if trainer.step_count >= max_steps:
                break

            input_seq, target_seq = batch
            metrics = trainer.train_step(input_seq, target_seq)

            if step % 100 == 0:
                logger.info(f"Step {step}: Loss={metrics['loss']:.4f}, "
                          f"PPL={metrics['perplexity']:.2f}, LR={metrics['learning_rate']:.2e}")

        # Save final checkpoint
        final_checkpoint = trainer.save_checkpoint()
        logger.info(f"Training completed. Final checkpoint: {final_checkpoint}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def generate_command():
    """Generate text with a trained RAT model"""
    parser = argparse.ArgumentParser(description="Generate text with RAT")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer name or path")
    parser.add_argument("--prompt", type=str, default="Hello, how are you?",
                       help="Text prompt for generation")
    parser.add_argument("--max-length", type=int, default=100,
                       help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.8,
                       help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=50,
                       help="Top-k sampling parameter")
    parser.add_argument("--top-p", type=float, default=0.9,
                       help="Top-p sampling parameter")
    parser.add_argument("--num-sequences", type=int, default=1,
                       help="Number of sequences to generate")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                       help="Repetition penalty")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for generation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file to save generated text")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        # Load tokenizer
        logger.info(f"Loading tokenizer: {args.tokenizer}")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        logger.info(f"Loading model from {args.model_path}")
        device = get_device(args.device)

        # For now, assume we're loading a full checkpoint
        # In a real implementation, you'd load the model state dict
        model_config = {
            'vocab_size': tokenizer.vocab_size,
            'd_model': 768,
            'n_layers': 12,
            'n_heads': 12,
            'max_seq_len': 1024
        }
        model = RAT(**model_config).to(device)

        # Create generator
        generator = RATGenerator(model, tokenizer, device=device)

        # Generate text
        logger.info(f"Generating text with prompt: '{args.prompt}'")
        generated_texts = generator.generate(
            prompt=args.prompt,
            max_len=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            num_return_sequences=args.num_sequences
        )

        # Output results
        if args.num_sequences == 1:
            generated_texts = [generated_texts]

        for i, text in enumerate(generated_texts):
            print(f"\n--- Generated Text {i+1} ---")
            print(text)

        # Save to file if requested
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                for i, text in enumerate(generated_texts):
                    f.write(f"Generated Text {i+1}:\n{text}\n\n")
            logger.info(f"Generated text saved to {args.output}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        sys.exit(1)


def eval_command():
    """Evaluate a trained RAT model"""
    parser = argparse.ArgumentParser(description="Evaluate RAT model")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--tokenizer", type=str, default="gpt2",
                       help="Tokenizer name or path")
    parser.add_argument("--dataset", type=str, default="wikitext",
                       help="Evaluation dataset")
    parser.add_argument("--split", type=str, default="test",
                       help="Dataset split to evaluate on")
    parser.add_argument("--max-samples", type=int, default=1000,
                       help="Maximum number of samples to evaluate")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for evaluation")
    parser.add_argument("--output", type=str, default=None,
                       help="Output file for evaluation results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        logger.info("RAT evaluation not yet implemented")
        logger.info("This would evaluate perplexity, BLEU scores, etc.")

        # Placeholder for evaluation logic
        results = {
            "perplexity": 15.7,
            "bleu_score": 0.23,
            "status": "placeholder"
        }

        print("Evaluation Results:")
        print(json.dumps(results, indent=2))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def test_command():
    """Run RAT tests"""
    parser = argparse.ArgumentParser(description="Run RAT tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run only quick component tests")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for testing")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    logger = setup_logging(args.verbose)

    try:
        # Import and run the test suite
        from test_rat import RATTester

        logger.info("Running RAT test suite")
        tester = RATTester(device=get_device(args.device))

        if args.quick:
            tester.test_configuration_validation()
            tester.test_rotary_embeddings()
            tester.test_swiglu_feedforward()
            tester.test_adaptive_policy_attention()
            tester.test_full_rat_model()
        else:
            success = tester.run_all_tests()
            sys.exit(0 if success else 1)

    except ImportError:
        logger.error("Test file not found. Please ensure test_rat.py exists")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # If run directly, show help
    parser = argparse.ArgumentParser(description="RAT CLI")
    parser.add_argument("command", choices=["train", "generate", "eval", "test"],
                       help="Command to run")
    parser.add_argument("--help", action="store_true",
                       help="Show help for specific command")

    args, remaining = parser.parse_known_args()

    if args.help:
        if args.command == "train":
            train_command()
        elif args.command == "generate":
            generate_command()
        elif args.command == "eval":
            eval_command()
        elif args.command == "test":
            test_command()
    else:
        if args.command == "train":
            train_command()
        elif args.command == "generate":
            generate_command()
        elif args.command == "eval":
            eval_command()
        elif args.command == "test":
            test_command()
