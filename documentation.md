# RAT: Reinforced Adaptive Transformer Documentation

## 📖 Overview

**RAT (Reinforced Adaptive Transformer)** is a next-generation transformer architecture that combines adaptive attention mechanisms powered by reinforcement learning with cutting-edge techniques like Rotary Position Embeddings, SwiGLU feed-forward networks, and temporal convolutions.

### Key Features

- 🧠 **Adaptive Policy Attention**: Dynamic head gating using multiple RL-based policy networks
- 🔄 **Rotary Position Embeddings**: Enhanced positional understanding with RoPE
- 🚀 **SwiGLU Feed-Forward**: Efficient activation for better expressiveness
- ⏰ **Temporal Convolutions**: Sequence modeling with depthwise convolutions
- 📊 **Advanced Logging**: Comprehensive training monitoring and debugging
- 🛡️ **Error Handling**: Robust validation and graceful failure recovery
- 💾 **Auto-Checkpointing**: Automatic model saving with training state

---

## 🚀 Installation

### From PyPI (Recommended)
```bash
pip install rat-transformer
```

### From Source
```bash
git clone https://github.com/arjun988/RAT.git
cd RAT
pip install -e .
```

### Optional Dependencies
```bash
# For development
pip install -e ".[dev]"

# For training with experiment tracking
pip install -e ".[training]"

# For model serving
pip install -e ".[serving]"
```

---

## 📚 API Reference

### Core Model Components

#### `RAT`

The main transformer model class implementing the Reinforced Adaptive Transformer architecture.

```python
class RAT(nn.Module):
    """
    RAT: Reinforced Adaptive Transformer

    A next-generation transformer architecture featuring:
    - Adaptive Policy Attention with multiple reinforcement learning policies
    - Rotary Position Embeddings for enhanced positional understanding
    - SwiGLU Feed-Forward networks for better expressiveness
    - Temporal convolution layers for improved sequence modeling
    """
```

**Parameters:**
- `vocab_size` (int): Size of token vocabulary
- `d_model` (int, default=768): Model dimension (must be divisible by n_heads)
- `n_layers` (int, default=12): Number of transformer layers
- `n_heads` (int, default=12): Number of attention heads
- `n_policies` (int, default=3): Number of RL policies for attention gating
- `max_seq_len` (int, default=2048): Maximum sequence length
- `dropout` (float, default=0.1): Dropout probability
- `mlp_ratio` (int, default=4): Feed-forward network expansion ratio
- `use_rope` (bool, default=True): Whether to use rotary position embeddings
- `tie_weights` (bool, default=True): Whether to tie input/output embeddings
- `use_checkpointing` (bool, default=False): Whether to use gradient checkpointing

**Methods:**

```python
def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
            past_kvs: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
            use_cache: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]]:
    """
    Forward pass through RAT

    Args:
        x: Input token ids [batch_size, seq_len]
        mask: Attention mask [batch_size, seq_len]
        past_kvs: Past key/value caches for generation
        use_cache: Whether to return KV caches for faster generation

    Returns:
        logits or (logits, kv_caches) if use_cache=True
    """
```

**Example:**
```python
import torch
from rat import RAT

# Create a small RAT model
model = RAT(
    vocab_size=30000,
    d_model=512,
    n_layers=6,
    n_heads=8,
    n_policies=3
)

# Forward pass
input_ids = torch.randint(0, 30000, (2, 128))  # batch_size=2, seq_len=128
logits = model(input_ids)  # Shape: [2, 128, 30000]
```

---

#### `AdaptivePolicyAttention`

Multi-head attention with reinforcement learning-based dynamic head gating.

```python
class AdaptivePolicyAttention(nn.Module):
    """
    Adaptive Policy Attention with Reinforcement Learning

    Implements dynamic head gating using multiple RL-based policy networks
    that adaptively control attention head contributions based on input context.
    """
```

**Parameters:**
- `d_model` (int): Model dimension
- `n_heads` (int): Number of attention heads
- `n_policies` (int, default=3): Number of RL policies
- `dropout` (float, default=0.1): Dropout probability
- `use_rope` (bool, default=True): Whether to use rotary position embeddings

**Example:**
```python
from rat import AdaptivePolicyAttention
import torch

attention = AdaptivePolicyAttention(
    d_model=512,
    n_heads=8,
    n_policies=3
)

# Forward pass
x = torch.randn(2, 128, 512)  # [batch, seq_len, d_model]
output = attention(x)  # [batch, seq_len, d_model]
```

---

#### `RATBlock`

Complete transformer block combining attention, feed-forward, and temporal convolution.

```python
class RATBlock(nn.Module):
    """
    RAT Transformer Block

    Combines adaptive policy attention, SwiGLU feed-forward networks,
    and temporal convolution for enhanced sequence modeling.
    """
```

**Parameters:**
- `d_model` (int): Model dimension
- `n_heads` (int): Number of attention heads
- `n_policies` (int, default=3): Number of RL policies
- `dropout` (float, default=0.1): Dropout probability
- `use_rope` (bool, default=True): Whether to use rotary position embeddings
- `mlp_ratio` (int, default=4): Feed-forward expansion ratio
- `use_checkpointing` (bool, default=False): Whether to use gradient checkpointing

**Example:**
```python
from rat import RATBlock
import torch

block = RATBlock(
    d_model=512,
    n_heads=8,
    n_policies=3
)

x = torch.randn(2, 128, 512)
output, kv_cache = block(x, use_cache=True)
```

---

#### `SwiGLUFeedForward`

Efficient feed-forward network using SwiGLU activation.

```python
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, multiple_of=256, dropout=0.1):
        # SwiGLU: FFN(x) = (xW1 ⊙ xW3)W2
        # More efficient than standard FFN
```

**Parameters:**
- `d_model` (int): Input/output dimension
- `d_ff` (int, optional): Hidden dimension (default: 4 * d_model)
- `multiple_of` (int, default=256): Hidden dimension rounding factor
- `dropout` (float, default=0.1): Dropout probability

**Example:**
```python
from rat import SwiGLUFeedForward
import torch

ffn = SwiGLUFeedForward(d_model=512, d_ff=2048)
x = torch.randn(2, 128, 512)
output = ffn(x)  # [2, 128, 512]
```

---

#### `RotaryPositionEmbedding`

Implements rotary position embeddings (RoPE) for enhanced positional understanding.

```python
class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        # RoPE enables better length extrapolation than absolute position embeddings
```

**Parameters:**
- `dim` (int): Embedding dimension (head_dim)
- `max_seq_len` (int, default=2048): Maximum sequence length

**Example:**
```python
from rat import RotaryPositionEmbedding, apply_rotary_pos_emb
import torch

rope = RotaryPositionEmbedding(dim=64, max_seq_len=1024)

# In attention mechanism
q = torch.randn(2, 8, 128, 64)  # [batch, heads, seq_len, head_dim]
k = torch.randn(2, 8, 128, 64)
q_rotated, k_rotated = apply_rotary_pos_emb(q, k, *rope(q, 128))
```

---

### Training and Inference

#### `RATTrainer`

Advanced trainer for RAT models with comprehensive logging and optimization.

```python
class RATTrainer:
    """
    Advanced trainer for RAT models with comprehensive logging,
    checkpointing, and optimization strategies.
    """
```

**Parameters:**
- `model` (nn.Module): RAT model to train
- `tokenizer`: Tokenizer for text processing
- `lr` (float, default=5e-4): Learning rate
- `warmup_steps` (int, default=1000): Learning rate warmup steps
- `max_steps` (int, default=100000): Total training steps
- `weight_decay` (float, default=0.1): Weight decay for regularization
- `grad_clip` (float, default=1.0): Gradient clipping threshold
- `device` (torch.device, optional): Training device
- `accum_steps` (int, default=1): Gradient accumulation steps
- `checkpoint_dir` (str, default="./checkpoints"): Checkpoint directory
- `log_level` (int, default=20): Logging level

**Methods:**

```python
def train_step(self, input_seq: torch.Tensor, target_seq: torch.Tensor) -> Dict[str, float]:
    """Perform one training step"""
    # Returns: {"loss": float, "learning_rate": float, "step": int, "perplexity": float, ...}

def save_checkpoint(self, filepath: Optional[str] = None) -> str:
    """Save training checkpoint"""

def load_checkpoint(self, filepath: str) -> None:
    """Load training checkpoint"""

def get_training_stats(self) -> Dict[str, Any]:
    """Get comprehensive training statistics"""
```

**Example:**
```python
from rat import RAT, RATTrainer
from transformers import AutoTokenizer
import torch

# Setup
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = RAT(vocab_size=tokenizer.vocab_size)

trainer = RATTrainer(
    model=model,
    tokenizer=tokenizer,
    lr=1e-4,
    max_steps=10000,
    device="cuda"
)

# Training loop
for step in range(100):
    input_seq = torch.randint(0, tokenizer.vocab_size, (4, 512))
    target_seq = torch.randint(0, tokenizer.vocab_size, (4, 512))

    metrics = trainer.train_step(input_seq, target_seq)
    print(f"Step {step}: Loss={metrics['loss']:.4f}, PPL={metrics['perplexity']:.2f}")

# Save checkpoint
trainer.save_checkpoint("rat_checkpoint.pt")
```

---

#### `RATGenerator`

Advanced text generation with multiple sampling strategies.

```python
class RATGenerator:
    """
    Advanced text generation for RAT models with multiple sampling strategies
    and comprehensive logging.
    """
```

**Parameters:**
- `model`: Trained RAT model
- `tokenizer`: Tokenizer for text encoding/decoding
- `device` (torch.device, optional): Device for generation
- `log_level` (int, default=20): Logging verbosity level

**Methods:**

```python
def generate(self, prompt: str, max_len: int = 100, temperature: float = 0.8,
            top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1,
            num_return_sequences: int = 1, do_sample: bool = True,
            pad_token_id: Optional[int] = None) -> Union[str, list]:
    """Generate text with advanced sampling strategies"""
```

**Example:**
```python
from rat import RAT, RATGenerator
from transformers import AutoTokenizer

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = RAT(vocab_size=tokenizer.vocab_size)  # Load trained weights

# Create generator
generator = RATGenerator(model, tokenizer, device="cuda")

# Generate text
prompt = "The future of AI is"
generated_text = generator.generate(
    prompt=prompt,
    max_len=50,
    temperature=0.8,
    top_k=40,
    top_p=0.9,
    repetition_penalty=1.1
)

print(f"Prompt: {prompt}")
print(f"Generated: {generated_text}")
```

---

### Data Handling

#### `RATDataset`

Enhanced dataset class for RAT training with preprocessing and validation.

```python
class RATDataset(Dataset):
    """
    Advanced dataset for RAT training with improved preprocessing
    and data validation.
    """
```

**Parameters:**
- `tokenized_texts` (List[List[int]]): List of tokenized text sequences
- `seq_len` (int, default=1024): Maximum sequence length
- `pad_token_id` (int, default=0): Token ID for padding
- `eos_token_id` (int, optional): End of sequence token ID

**Methods:**

```python
def get_statistics(self) -> Dict[str, Any]:
    """Get dataset statistics"""
    # Returns: {"num_sequences": int, "avg_length": float, "max_length": int, ...}
```

**Example:**
```python
from rat import RATDataset
import torch

# Create tokenized texts (normally from tokenizer)
tokenized_texts = [
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10,  # Long sequence
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],        # Shorter sequence
]

dataset = RATDataset(
    tokenized_texts=tokenized_texts,
    seq_len=64,
    pad_token_id=0
)

# Get statistics
stats = dataset.get_statistics()
print(f"Dataset: {stats['num_sequences']} sequences, avg length: {stats['avg_length']:.1f}")

# Use with DataLoader
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for input_ids, target_ids in dataloader:
    print(f"Input shape: {input_ids.shape}, Target shape: {target_ids.shape}")
    break
```

---

#### `RATDataLoader`

Enhanced data loader with memory optimization and progress tracking.

```python
class RATDataLoader:
    """
    Enhanced data loader with memory optimization and progress tracking
    """
```

**Parameters:**
- `dataset_name` (str, default="wikitext"): HuggingFace dataset name
- `dataset_config` (str, default="wikitext-2-raw-v1"): Dataset configuration
- `tokenizer`: Tokenizer for text processing
- `seq_len` (int, default=1024): Maximum sequence length
- `batch_size` (int, default=4): Batch size for training
- `num_samples` (int, default=5000): Number of training samples to create
- `max_texts` (int, default=2000): Maximum number of texts to process

**Methods:**

```python
def prepare_dataset(self) -> DataLoader:
    """Prepare and load the dataset"""
```

**Example:**
```python
from rat import RATDataLoader
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

data_loader = RATDataLoader(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer=tokenizer,
    seq_len=512,
    batch_size=8,
    num_samples=10000
)

# Prepare dataset (downloads and processes)
train_dataloader = data_loader.prepare_dataset()

print(f"Created dataset with {len(train_dataloader)} batches")

# Training loop
for batch in train_dataloader:
    input_ids, target_ids = batch
    print(f"Batch shape: {input_ids.shape}")
    break
```

---

### Utilities

#### `RATLogger`

Advanced logging system for RAT models.

```python
class RATLogger:
    """Advanced logging system for RAT models"""
```

**Parameters:**
- `name` (str, default="RAT"): Logger name
- `level` (int, default=logging.INFO): Logging level
- `log_file` (str, optional): Optional log file path

**Methods:**

```python
def debug(self, message: str): """Log debug message"""
def info(self, message: str): """Log info message"""
def warning(self, message: str): """Log warning message"""
def error(self, message: str): """Log error message"""
def critical(self, message: str): """Log critical message"""

def log_model_info(self, model, model_name: str = "RAT"):
    """Log comprehensive model information"""

def log_training_step(self, step: int, loss: float, lr: float, **metrics):
    """Log training step information"""

def log_memory_usage(self, device: torch.device):
    """Log current memory usage"""
```

**Example:**
```python
from rat import RATLogger, RAT

# Create logger
logger = RATLogger("MyRAT", log_file="training.log")

# Log model info
model = RAT(vocab_size=30000, d_model=512)
logger.log_model_info(model, "My Custom RAT")

# Log training steps
for step in range(100):
    loss = 3.5 - step * 0.01  # Simulated decreasing loss
    logger.log_training_step(step, loss, 1e-4, perplexity=27.1)
```

---

#### `validate_model_config`

Validate and sanitize model configuration.

```python
def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize model configuration

    Args:
        config: Model configuration dictionary

    Returns:
        Validated and sanitized configuration

    Raises:
        ValueError: If configuration is invalid
    """
```

**Example:**
```python
from rat import validate_model_config

config = {
    'vocab_size': 30000,
    'd_model': 768,
    'n_layers': 12,
    'n_heads': 12,
    'max_seq_len': 2048
}

try:
    validated_config = validate_model_config(config)
    print("Configuration is valid!")
except ValueError as e:
    print(f"Invalid configuration: {e}")
```

---

#### `ModelCheckpoint`

Advanced checkpoint management for RAT models.

```python
class ModelCheckpoint:
    """Advanced checkpoint management for RAT models"""
```

**Parameters:**
- `save_dir` (str, default="./checkpoints"): Checkpoint directory
- `max_checkpoints` (int, default=5): Maximum number of checkpoints to keep

**Methods:**

```python
def save(self, model, optimizer, scheduler, step: int, loss: float,
         additional_info: Dict[str, Any] = None) -> str:
    """Save model checkpoint with comprehensive metadata"""

def load(self, checkpoint_path: str, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
```

**Example:**
```python
from rat import ModelCheckpoint, RAT, RATTrainer
import torch

# Setup checkpoint manager
checkpoint_manager = ModelCheckpoint("./my_checkpoints", max_checkpoints=3)

# During training
model = RAT(vocab_size=30000)
trainer = RATTrainer(model, tokenizer=None)  # Simplified example

# Save checkpoint
checkpoint_path = checkpoint_manager.save(
    model=model,
    optimizer=trainer.optimizer,
    scheduler=trainer.scheduler,
    step=1000,
    loss=2.5,
    additional_info={"epoch": 5, "validation_loss": 2.8}
)

print(f"Checkpoint saved: {checkpoint_path}")

# Load checkpoint later
checkpoint_manager.load(checkpoint_path, model, trainer.optimizer, trainer.scheduler)
```

---

#### `get_optimal_batch_size`

Find optimal batch size based on available memory.

```python
def get_optimal_batch_size(model, seq_len: int, max_memory_gb: float = 8.0,
                          device: torch.device = None) -> int:
    """
    Find optimal batch size based on available memory

    Args:
        model: RAT model
        seq_len: Sequence length
        max_memory_gb: Maximum memory to use (GB)
        device: Target device

    Returns:
        Optimal batch size
    """
```

**Example:**
```python
from rat import RAT, get_optimal_batch_size

model = RAT(vocab_size=30000, d_model=1024, n_layers=24)

optimal_batch = get_optimal_batch_size(
    model=model,
    seq_len=512,
    max_memory_gb=8.0,  # Use up to 8GB
    device=torch.device("cuda")
)

print(f"Optimal batch size: {optimal_batch}")
```

---

## 🎯 Usage Examples

### Complete Training Pipeline

```python
import torch
from rat import RAT, RATTrainer, RATDataLoader
from transformers import AutoTokenizer

# 1. Setup tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = RAT(
    vocab_size=tokenizer.vocab_size,
    d_model=512,
    n_layers=8,
    n_heads=8,
    n_policies=3,
    max_seq_len=512
)

# 2. Prepare dataset
data_loader = RATDataLoader(
    dataset_name="wikitext",
    dataset_config="wikitext-2-raw-v1",
    tokenizer=tokenizer,
    seq_len=512,
    batch_size=4,
    num_samples=5000
)
train_dataloader = data_loader.prepare_dataset()

# 3. Setup trainer
trainer = RATTrainer(
    model=model,
    tokenizer=tokenizer,
    lr=1e-4,
    max_steps=10000,
    grad_clip=1.0,
    checkpoint_dir="./checkpoints"
)

# 4. Training loop
print("Starting training...")
for step, batch in enumerate(train_dataloader):
    if trainer.step_count >= trainer.max_steps:
        break

    input_seq, target_seq = batch
    metrics = trainer.train_step(input_seq, target_seq)

    if step % 100 == 0:
        print(f"Step {step}: Loss={metrics['loss']:.4f}, PPL={metrics['perplexity']:.2f}")

# 5. Save final model
trainer.save_checkpoint("final_rat_model.pt")
print("Training completed!")
```

### Text Generation

```python
from rat import RAT, RATGenerator
from transformers import AutoTokenizer

# Load trained model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = RAT(vocab_size=tokenizer.vocab_size)
# Load your trained weights here
# model.load_state_dict(torch.load("trained_rat.pt"))

# Create generator
generator = RATGenerator(model, tokenizer)

# Generate text with different strategies
prompt = "The future of artificial intelligence"

# Greedy decoding
greedy_text = generator.generate(prompt, max_len=50, do_sample=False)
print("Greedy:", greedy_text)

# Sampling with temperature
sampled_text = generator.generate(
    prompt,
    max_len=50,
    temperature=0.8,
    top_k=40,
    top_p=0.9
)
print("Sampled:", sampled_text)

# Multiple sequences
multiple_texts = generator.generate(
    prompt,
    max_len=30,
    num_return_sequences=3,
    temperature=1.0
)
for i, text in enumerate(multiple_texts):
    print(f"Variant {i+1}:", text)
```

### Custom Model Configuration

```python
from rat import RAT, validate_model_config

# Define custom configuration
config = {
    'vocab_size': 50000,      # Large vocabulary
    'd_model': 1024,          # Large model dimension
    'n_layers': 24,           # Many layers
    'n_heads': 16,            # Many attention heads
    'n_policies': 5,          # Multiple RL policies
    'max_seq_len': 4096,      # Long sequences
    'dropout': 0.05,          # Low dropout for large model
    'mlp_ratio': 4,           # Standard FFN ratio
    'use_rope': True,         # Use RoPE
    'tie_weights': True,      # Tie embeddings
    'use_checkpointing': True # Use gradient checkpointing for memory
}

# Validate configuration
validated_config = validate_model_config(config)

# Create model
model = RAT(**validated_config)
print(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")
```

---

## 🔧 CLI Tools

RAT provides command-line tools for common tasks:

```bash
# Train a model
rat-train --config config.json --output-dir ./checkpoints

# Generate text
rat-generate --model-path checkpoints/model.pt --prompt "Hello world"

# Run tests
rat-test --quick

# Evaluate model
rat-eval --model-path model.pt --dataset wikitext
```

### Configuration File Example

```json
{
  "model": {
    "vocab_size": 30000,
    "d_model": 768,
    "n_layers": 12,
    "n_heads": 12,
    "n_policies": 3,
    "max_seq_len": 2048,
    "dropout": 0.1,
    "use_rope": true,
    "tie_weights": true
  },
  "trainer": {
    "lr": 0.0005,
    "warmup_steps": 1000,
    "max_steps": 100000,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "accum_steps": 4
  },
  "dataset": {
    "dataset_name": "wikitext",
    "dataset_config": "wikitext-2-raw-v1",
    "seq_len": 1024,
    "batch_size": 8,
    "num_samples": 10000
  },
  "tokenizer_path": "gpt2"
}
```

---

## 📊 Model Architectures

### Small Model (for testing)
```python
small_rat = RAT(
    vocab_size=10000,
    d_model=256,
    n_layers=4,
    n_heads=4,
    n_policies=2,
    max_seq_len=512
)
# ~3M parameters
```

### Base Model (recommended for most tasks)
```python
base_rat = RAT(
    vocab_size=30000,
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_policies=3,
    max_seq_len=2048
)
# ~85M parameters
```

### Large Model (for advanced tasks)
```python
large_rat = RAT(
    vocab_size=50000,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_policies=4,
    max_seq_len=4096
)
# ~340M parameters
```

---

## 🔍 Troubleshooting

### Common Issues

1. **CUDA out of memory**
   ```python
   # Reduce batch size or model size
   model = RAT(d_model=512, n_layers=8)  # Smaller model
   trainer = RATTrainer(accum_steps=4)  # Gradient accumulation
   ```

2. **Invalid configuration**
   ```python
   # Check that d_model is divisible by n_heads
   assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
   ```

3. **Training instability**
   ```python
   # Use gradient clipping and lower learning rate
   trainer = RATTrainer(grad_clip=0.5, lr=1e-4)
   ```

### Memory Optimization

```python
# Enable gradient checkpointing for large models
model = RAT(use_checkpointing=True)

# Use mixed precision (if available)
# trainer = RATTrainer(use_amp=True)  # Future feature

# Monitor memory usage
trainer.logger.log_memory_usage(trainer.device)
```

---

## 📈 Performance Tips

1. **Use appropriate batch sizes**: Start small and increase gradually
2. **Enable gradient checkpointing** for models >10M parameters
3. **Use RoPE** for better length extrapolation
4. **Tie weights** to reduce parameter count
5. **Monitor perplexity** as the key metric
6. **Use multiple policies** for complex tasks
7. **Implement early stopping** based on validation loss

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Ensure all tests pass
5. Update documentation
6. Submit a pull request

### Development Setup

```bash
git clone https://github.com/arjun988/RAT.git
cd RAT
pip install -e ".[dev]"
pytest test_rat.py
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE.txt file for details.

---

## 🙏 Acknowledgments

- Built on PyTorch and Hugging Face Transformers
- Inspired by modern transformer architectures
- Thanks to the research community for advancing RL and attention mechanisms

---

*RAT: Reinventing transformers with reinforcement learning*
