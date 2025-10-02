import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm

class GPTTextDataset(Dataset):
    def __init__(self, tokenized_texts, seq_len=1024):
        self.seq_len = seq_len
        self.tokenized_texts = tokenized_texts
    def __len__(self):
        return len(self.tokenized_texts)
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        if len(tokens) > self.seq_len + 1:
            start_idx = torch.randint(0, len(tokens) - self.seq_len - 1, (1,)).item()
            tokens = tokens[start_idx:start_idx + self.seq_len + 1]
        elif len(tokens) < self.seq_len + 1:
            tokens = tokens + [0] * (self.seq_len + 1 - len(tokens))
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        return input_ids, target_ids

def prepare_gpt_dataset(tokenizer, seq_len=1024, batch_size=4, num_samples=5000):
    print("Loading and preparing GPT-style dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    train_texts = [text for text in dataset["train"]["text"] if text.strip()]
    print(f"Loaded {len(train_texts)} training texts")
    all_tokens = []
    max_chunk_length = seq_len * 2
    for text in tqdm(train_texts[:2000]):
        if text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            for i in range(0, len(tokens), max_chunk_length):
                chunk = tokens[i:i + max_chunk_length]
                if len(chunk) >= 10:
                    all_tokens.extend(chunk)
    print(f"Total tokens: {len(all_tokens):,}")
    sequences = []
    for i in range(0, len(all_tokens) - seq_len, seq_len):
        sequences.append(all_tokens[i:i + seq_len + 1])
        if len(sequences) >= num_samples:
            break
    train_dataset = GPTTextDataset(sequences, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print(f"Created {len(sequences)} training sequences")
    return train_loader
