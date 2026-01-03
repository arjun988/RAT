import torch
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any
import time
from rat_model.utils import RATLogger, safe_tensor_operation, rat_logger


class RATGenerator:
    """
    Advanced text generation for RAT models with multiple sampling strategies
    and comprehensive logging.
    """

    def __init__(self, model, tokenizer, device: Optional[torch.device] = None,
                 log_level: int = 20):
        """
        Initialize NeuroForge generator

        Args:
            model: Trained RAT model
            tokenizer: Tokenizer for text encoding/decoding
            device: Device for generation
            log_level: Logging verbosity level
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = RATLogger("RATGenerator", level=log_level)

        # Validate tokenizer
        if not hasattr(tokenizer, 'eos_token_id') or tokenizer.eos_token_id is None:
            self.logger.warning("Tokenizer missing eos_token_id, setting to 2")
            tokenizer.eos_token_id = 2

        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.logger.info(f"Generator initialized on device: {self.device}")
        self.logger.debug(f"Model vocab size: {getattr(model, 'vocab_size', 'unknown')}")

    @torch.no_grad()
    @safe_tensor_operation
    def generate(self, prompt: str, max_len: int = 100, temperature: float = 0.8,
                top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1,
                num_return_sequences: int = 1, do_sample: bool = True,
                pad_token_id: Optional[int] = None) -> Union[str, list]:
        """
        Generate text using RAT model

        Args:
            prompt: Input text prompt
            max_len: Maximum length of generated sequence
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Penalty for repeating tokens
            num_return_sequences: Number of sequences to generate
            do_sample: Whether to use sampling (vs greedy decoding)
            pad_token_id: Token ID for padding

        Returns:
            Generated text or list of texts
        """
        try:
            start_time = time.time()

            # Validate inputs
            if not isinstance(prompt, str) or len(prompt.strip()) == 0:
                raise ValueError("Prompt must be a non-empty string")

            if temperature <= 0:
                raise ValueError(f"Temperature must be positive, got {temperature}")

            if max_len <= 0:
                raise ValueError(f"max_len must be positive, got {max_len}")

            self.model.eval()

            # Tokenize input
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs["input_ids"].to(self.device)
            except Exception as e:
                raise ValueError(f"Tokenization failed: {e}")

            batch_size = input_ids.shape[0]
            generated_sequences = []

            for seq_idx in range(num_return_sequences):
                generated = input_ids.clone()
                past_kvs = None
                generation_stats = {
                    'tokens_generated': 0,
                    'eos_reached': False,
                    'repetition_penalty_applied': 0
                }

                for step in range(max_len):
                    try:
                        # Forward pass with KV caching
                        if past_kvs is not None:
                            # Use only the last token for efficiency
                            current_input = generated[:, -1:]
                        else:
                            current_input = generated

                        logits, past_kvs = self.model(current_input, past_kvs=past_kvs, use_cache=True)
                        next_token_logits = logits[:, -1, :]

                        # Apply temperature
                        if do_sample:
                            next_token_logits = next_token_logits / max(temperature, 1e-8)

                        # Apply repetition penalty
                        if repetition_penalty != 1.0:
                            seen_tokens = set(generated[0].tolist())
                            for token_id in seen_tokens:
                                if token_id < next_token_logits.shape[-1]:
                                    next_token_logits[0, token_id] /= repetition_penalty
                                    generation_stats['repetition_penalty_applied'] += 1

                        # Apply top-k sampling
                        if top_k > 0 and do_sample:
                            top_k_val = min(top_k, next_token_logits.shape[-1])
                            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                            next_token_logits[indices_to_remove] = -float("inf")

                        # Apply top-p (nucleus) sampling
                        if top_p < 1.0 and do_sample:
                            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                            # Remove tokens with cumulative probability above the threshold
                            sorted_indices_to_remove = cumulative_probs > top_p
                            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                            sorted_indices_to_remove[..., 0] = 0

                            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                            next_token_logits[indices_to_remove] = -float("inf")

                        # Sample next token
                        if do_sample:
                            probs = F.softmax(next_token_logits, dim=-1)
                            next_token = torch.multinomial(probs, num_samples=1)
                        else:
                            # Greedy decoding
                            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                        # Append to sequence
                        generated = torch.cat([generated, next_token], dim=1)
                        generation_stats['tokens_generated'] += 1

                        # Check for EOS token
                        if next_token.item() == self.tokenizer.eos_token_id:
                            generation_stats['eos_reached'] = True
                            break

                    except Exception as e:
                        self.logger.error(f"Generation step {step} failed: {e}")
                        break

                # Decode generated sequence
                try:
                    generated_text = self.tokenizer.decode(
                        generated[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=True
                    )
                    generated_sequences.append(generated_text)
                except Exception as e:
                    self.logger.error(f"Decoding failed: {e}")
                    generated_sequences.append("[DECODE_ERROR]")

            generation_time = time.time() - start_time
            tokens_per_sec = sum(len(seq.split()) for seq in generated_sequences) / generation_time

            # Log generation statistics
            self.logger.log_generation_stats(prompt, generated_sequences[0], generation_time)
            self.logger.debug(f"Generation stats: {generation_stats}")
            self.logger.debug(".2f")

            if num_return_sequences == 1:
                return generated_sequences[0]
            return generated_sequences

        except Exception as e:
            self.logger.error(f"Text generation failed: {e}")
            raise RuntimeError(f"RAT generation failed: {e}") from e


# Backward compatibility function
@torch.no_grad()
def gpt_generate(model, tokenizer, prompt, max_len=100, temperature=0.8, top_k=50,
                top_p=0.9, repetition_penalty=1.1, device=None):
    """
    Legacy function for backward compatibility.
    Use RATGenerator class for new code.
    """
    generator = RATGenerator(model, tokenizer, device)
    return generator.generate(
        prompt=prompt,
        max_len=max_len,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
