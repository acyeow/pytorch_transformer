# PyTorch Transformer for Neural Machine Translation

![Transformer Architecture](https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png)

## 🚀 Project Overview

This project implements a Transformer architecture from scratch using PyTorch for neural machine translation tasks. The implementation closely follows the architecture described in the "Attention is All You Need" paper by Vaswani et al. (2017), featuring multi-head attention mechanisms, positional encodings, and encoder-decoder structure.

## ✨ Key Features

- **Full Transformer Implementation**: Built from the ground up including all essential components
- **Multi-head Attention**: Implementation of scaled dot-product attention with multiple heads
- **Positional Encodings**: Sinusoidal position embeddings to incorporate sequence order
- **Layer Normalization & Residual Connections**: For stable and efficient training
- **Greedy Decoding**: For inference and translation generation
- **Tensorboard Integration**: For monitoring training metrics

## 🔧 Technical Architecture

### Model Components

The transformer is built with the following components:

| Component                | Description                                                                                 |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| **Encoder**              | Stack of N identical layers with self-attention and feed-forward networks                   |
| **Decoder**              | Stack of N identical layers with self-attention, cross-attention, and feed-forward networks |
| **Multi-Head Attention** | Allows the model to focus on different positions in parallel                                |
| **Position-wise FFN**    | Two linear transformations with a ReLU activation in between                                |
| **Layer Normalization**  | Applied to stabilize training                                                               |
| **Residual Connections** | Applied around each sub-layer to facilitate gradient flow                                   |

### Implementation Details

- **Model Dimensions**: d_model=512, heads=8, layers=6 (configurable)
- **Tokenization**: Word-level tokenization using HuggingFace's tokenizers library
- **Dataset**: OPUS Books corpus for English-Italian translation
- **Training**: Adam optimizer with learning rate of 1e-4
- **Regularization**: Dropout of 0.1 and label smoothing of 0.1

## 📊 Performance and Metrics

The model is trained on the OPUS Books dataset for English to Italian translation. During training, validation is performed to monitor:

- Translation quality
- Cross-entropy loss
- Model convergence

## 🧠 Technical Challenges Solved

- **Memory Efficiency**: Implemented efficient attention mechanisms with proper masking
- **Training Stability**: Used layer normalization and residual connections for stable gradient flow
- **Customizable Architecture**: Designed components to be modular and configurable
- **Tokenization**: Built custom tokenizers for source and target languages

## 📈 Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/pytorch_transformer.git
cd pytorch_transformer

# Install dependencies
pip install torch torchvision torchaudio
pip install datasets tokenizers tqdm
```

### Training

```bash
python train.py
```

This will start the training process using the configuration specified in `config.py`.

### Configuration

Edit `config.py` to change model parameters:

```python
{
    "batch_size": 8,
    "num_epochs": 20,
    "lr": 10**-4,
    "seq_len": 350,
    "d_model": 512,
    "datasource": 'opus_books',
    "lang_src": "en",
    "lang_tgt": "it",
    # ...other parameters
}
```

## 🛠️ Project Structure

```
pytorch_transformer/
├── train.py         # Training logic and validation
├── model.py         # Transformer model implementation
├── dataset.py       # Dataset preparation and loading
├── config.py        # Configuration parameters
└── .gitignore       # Git ignore file
```

## 📚 Key Learning Outcomes

- Deep understanding of Transformer architecture and attention mechanisms
- Implementation of complex deep learning models from scratch
- Efficient data preprocessing for NLP tasks
- Training and optimization of large neural network models
- Practical application of advanced PyTorch features

## 🔗 Additional Resources

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original paper
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

---

### Acknowledgments

This implementation draws inspiration from the original Transformer paper and various open-source implementations in the deep learning community.
