# RAT: Reinforced Adaptive Transformer

<p align="center">
  <img src="assests/Architechture.png" alt="RAT Architecture" width="400" height="800"/>
</p>

RAT (Reinforced Adaptive Transformer) is a next-generation transformer architecture featuring adaptive attention mechanisms powered by reinforcement learning. It combines cutting-edge techniques like Rotary Position Embeddings, SwiGLU feed-forward networks, and temporal convolutions for superior language modeling performance.

## ✨ Key Features

- **🧠 Adaptive Policy Attention**: Dynamic head gating using multiple RL-based policy networks
- **🔄 Rotary Position Embeddings**: Enhanced positional understanding with RoPE
- **🚀 SwiGLU Feed-Forward**: Efficient activation for better expressiveness
- **⏰ Temporal Convolutions**: Sequence modeling with depthwise convolutions
- **📊 Advanced Logging**: Comprehensive training monitoring and debugging
- **🛡️ Error Handling**: Robust validation and graceful failure recovery
- **💾 Auto-Checkpointing**: Automatic model saving with training state
- **🎯 Optimized Generation**: Multiple sampling strategies with KV caching

## 🏗️ Architecture Components

### Core Components
- **`RAT`**: Main transformer model with adaptive attention
- **`AdaptivePolicyAttention`**: Multi-policy attention with reinforcement learning
- **`RATBlock`**: Transformer block with attention, FFN, and temporal conv
- **`SwiGLUFeedForward`**: Efficient feed-forward network
- **`RotaryPositionEmbedding`**: Rotary positional encodings

### Training & Inference
- **`RATTrainer`**: Advanced trainer with logging and checkpointing
- **`RATGenerator`**: Optimized text generation with multiple strategies
- **`RATDataset`**: Enhanced dataset with preprocessing and validation

### Utilities
- **`RATLogger`**: Comprehensive logging system
- **`ModelCheckpoint`**: Automatic checkpoint management
- **Configuration validation**: Input sanitization and error checking

## 🚀 Quick Start

### Installation

#### From PyPI (Recommended)
```bash
pip install rat-transformer
```

#### From Source
```bash
# Clone the repository
git clone https://github.com/ReinforcedAdaptiveTransformer-RAT/RAT.git
cd RAT

# Install in development mode
pip install -e .

# Or install with optional dependencies
pip install -e ".[dev,training,serving]"
```

### Basic Usage

```python
from rat import RAT, RATTrainer, RATGenerator
from transformers import AutoTokenizer

# Initialize model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = RAT(vocab_size=tokenizer.vocab_size)

# Training
trainer = RATTrainer(model, tokenizer)
# ... training code ...

# Generation
generator = RATGenerator(model, tokenizer)
text = generator.generate("Hello, how are you?", max_len=50)
print(text)
```

#### Command Line Interface

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

### Advanced Configuration

```python
# Custom model configuration
model = RAT(
    vocab_size=50000,
    d_model=1024,
    n_layers=24,
    n_heads=16,
    n_policies=5,
    dropout=0.1,
    use_rope=True,
    use_checkpointing=True
)

# Advanced training
trainer = RATTrainer(
    model=model,
    tokenizer=tokenizer,
    lr=1e-4,
    max_steps=100000,
    grad_clip=1.0,
    checkpoint_dir="./checkpoints"
)
```

## 📊 Performance & Benchmarks

- **Parameter Efficiency**: Better performance with fewer parameters
- **Training Stability**: Advanced optimization and regularization
- **Generation Quality**: Superior text coherence and diversity
- **Memory Optimization**: Gradient checkpointing and KV caching

## 🔧 Configuration

### Model Parameters
- `vocab_size`: Size of token vocabulary
- `d_model`: Model dimension (must be divisible by n_heads)
- `n_layers`: Number of transformer layers
- `n_heads`: Number of attention heads
- `n_policies`: Number of RL policies for attention gating
- `max_seq_len`: Maximum sequence length
- `dropout`: Dropout probability

### Training Parameters
- `lr`: Learning rate
- `warmup_steps`: Learning rate warmup steps
- `weight_decay`: Weight decay for regularization
- `grad_clip`: Gradient clipping threshold
- `accum_steps`: Gradient accumulation steps

## 🧪 Testing & Validation

Run the comprehensive test suite:

```bash
python test_rat.py
```

The test suite validates:
- ✅ Component functionality
- ✅ Training pipeline
- ✅ Text generation
- ✅ Memory usage
- ✅ Gradient flow
- ✅ Error handling

## 📈 Training Tips

1. **Batch Size**: Start with smaller batches and increase gradually
2. **Learning Rate**: Use 1e-4 for large models, 5e-4 for smaller ones
3. **Gradient Accumulation**: Use for effective larger batch sizes
4. **Checkpointing**: Enable automatic saving every 1000 steps
5. **Monitoring**: Watch perplexity and loss curves

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Inspired by modern transformer architectures
- Built on PyTorch and Hugging Face Transformers
- Thanks to the research community for advancing transformer models

---

*RAT: Reinforced Adaptive Transformer - Revolutionizing language models with reinforcement learning*
