import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from rat_model.utils import RATLogger, rat_logger


class RATDataset(Dataset):
    """
    Advanced dataset for RAT training with improved preprocessing
    and data validation.
    """

    def __init__(self, tokenized_texts: List[List[int]], seq_len: int = 1024,
                 pad_token_id: int = 0, eos_token_id: Optional[int] = None):
        """
        Initialize RAT dataset

        Args:
            tokenized_texts: List of tokenized text sequences
            seq_len: Maximum sequence length
            pad_token_id: Token ID for padding
            eos_token_id: End of sequence token ID
        """
        if not tokenized_texts:
            raise ValueError("tokenized_texts cannot be empty")

        self.seq_len = seq_len
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Validate and filter sequences
        valid_sequences = []
        for i, tokens in enumerate(tokenized_texts):
            if len(tokens) >= 10:  # Minimum sequence length
                valid_sequences.append(tokens)
            elif len(tokenized_texts) < 100:  # Only warn for small datasets
                rat_logger.warning(f"Skipping sequence {i} with length {len(tokens)}")

        if not valid_sequences:
            raise ValueError("No valid sequences found after filtering")

        self.tokenized_texts = valid_sequences

        rat_logger.info(f"RATDataset initialized with {len(valid_sequences)} sequences, "
                              f"max_seq_len={seq_len}")

    def __len__(self) -> int:
        return len(self.tokenized_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample

        Args:
            idx: Sample index

        Returns:
            Tuple of (input_ids, target_ids)
        """
        try:
            tokens = self.tokenized_texts[idx]

            # Handle sequences longer than seq_len + 1
            if len(tokens) > self.seq_len + 1:
                # Random crop for training
                max_start = len(tokens) - self.seq_len - 1
                if max_start > 0:
                    start_idx = torch.randint(0, max_start, (1,)).item()
                else:
                    start_idx = 0
                tokens = tokens[start_idx:start_idx + self.seq_len + 1]

            # Handle sequences shorter than seq_len + 1
            elif len(tokens) < self.seq_len + 1:
                # Pad with pad_token_id
                padding_needed = self.seq_len + 1 - len(tokens)
                tokens = tokens + [self.pad_token_id] * padding_needed

            # Ensure we have exactly seq_len + 1 tokens
            tokens = tokens[:self.seq_len + 1]

            # Create input and target sequences
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)

            return input_ids, target_ids

        except Exception as e:
            rat_logger.error(f"Error in dataset __getitem__ at index {idx}: {e}")
            # Return zero tensors as fallback
            return torch.zeros(self.seq_len, dtype=torch.long), torch.zeros(self.seq_len, dtype=torch.long)

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        lengths = [len(seq) for seq in self.tokenized_texts]
        return {
            'num_sequences': len(self.tokenized_texts),
            'avg_length': np.mean(lengths),
            'max_length': max(lengths),
            'min_length': min(lengths),
            'total_tokens': sum(lengths)
        }


class RATDataLoader:
    """
    Enhanced data loader with memory optimization and progress tracking
    """

    def __init__(self, dataset_name: str = "wikitext", dataset_config: str = "wikitext-2-raw-v1",
                 tokenizer=None, seq_len: int = 1024, batch_size: int = 4,
                 num_samples: int = 5000, max_texts: int = 2000):
        """
        Initialize RAT data loader

        Args:
            dataset_name: HuggingFace dataset name
            dataset_config: Dataset configuration
            tokenizer: Tokenizer for text processing
            seq_len: Maximum sequence length
            batch_size: Batch size for training
            num_samples: Number of training samples to create
            max_texts: Maximum number of texts to process
        """
        self.logger = RATLogger("RATDataLoader")
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_texts = max_texts

        if tokenizer is None:
            raise ValueError("Tokenizer is required")

        self.tokenizer = tokenizer

    def prepare_dataset(self) -> DataLoader:
        """
        Prepare and load the dataset

        Returns:
            PyTorch DataLoader for training
        """
        try:
            self.logger.info(f"Loading dataset: {self.dataset_name} ({self.dataset_config})")

            # Load dataset
            dataset = load_dataset(self.dataset_name, self.dataset_config)

            # Extract and filter training texts
            train_texts = [text for text in dataset["train"]["text"] if text.strip()]

            if len(train_texts) > self.max_texts:
                train_texts = train_texts[:self.max_texts]
                self.logger.info(f"Limited to {self.max_texts} texts for processing")

            self.logger.info(f"Processing {len(train_texts)} training texts")

            # Tokenize texts with progress bar
            all_tokens = []
            max_chunk_length = self.seq_len * 2

            with tqdm(total=len(train_texts), desc="Tokenizing") as pbar:
                for text in train_texts:
                    try:
                        tokens = self.tokenizer.encode(text, add_special_tokens=False)

                        # Process in chunks to avoid memory issues
                        for i in range(0, len(tokens), max_chunk_length):
                            chunk = tokens[i:i + max_chunk_length]
                            if len(chunk) >= 10:  # Minimum chunk length
                                all_tokens.extend(chunk)

                    except Exception as e:
                        self.logger.warning(f"Failed to tokenize text: {e}")
                        continue

                    pbar.update(1)

            total_tokens = len(all_tokens)
            self.logger.info(f"Total tokens collected: {total_tokens:,}")

            if total_tokens < self.seq_len * 10:
                raise ValueError(f"Insufficient data: only {total_tokens} tokens collected")

            # Create training sequences
            sequences = []
            step_size = max(1, self.seq_len // 4)  # Overlapping sequences

            for i in range(0, len(all_tokens) - self.seq_len, step_size):
                sequence = all_tokens[i:i + self.seq_len + 1]
                sequences.append(sequence)

                if len(sequences) >= self.num_samples:
                    break

            self.logger.info(f"Created {len(sequences)} training sequences")

            if len(sequences) == 0:
                raise ValueError("No training sequences created")

            # Create dataset and dataloader
            dataset = RATDataset(
                sequences,
                seq_len=self.seq_len,
                pad_token_id=getattr(self.tokenizer, 'pad_token_id', 0),
                eos_token_id=getattr(self.tokenizer, 'eos_token_id', None)
            )

            # Log dataset statistics
            stats = dataset.get_statistics()
            self.logger.info("Dataset statistics:")
            for key, value in stats.items():
                self.logger.info(f"  {key}: {value}")

            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=0,  # Avoid multiprocessing issues
                pin_memory=torch.cuda.is_available(),
                drop_last=True  # Ensure consistent batch sizes
            )

            self.logger.info(f"DataLoader created with batch_size={self.batch_size}")
            return dataloader

        except Exception as e:
            self.logger.error(f"Dataset preparation failed: {e}")
            raise


# Backward compatibility functions
class GPTTextDataset(RATDataset):
    """Legacy class for backward compatibility"""
    pass


def prepare_gpt_dataset(tokenizer, seq_len=1024, batch_size=4, num_samples=5000):
    """Legacy function for backward compatibility"""
    loader = RATDataLoader(tokenizer=tokenizer, seq_len=seq_len,
                                 batch_size=batch_size, num_samples=num_samples)
    return loader.prepare_dataset()
