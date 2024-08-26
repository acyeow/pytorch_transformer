import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of size (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # Apply the sin to the even positions and cos to the odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # (1, seq_len, d_model)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :self.seq_len, :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, eps: float = 10e-6) -> None:
        super().__init__()
        self.eps = eps
        # Multiplicative
        self.alpha = nn.Parameter(torch.ones(1))
        # Additive
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        x = self.linear_1(x) # W1 and B1
        # Apply activation
        x = torch.relu(x)
        # Apply dropout
        x = torch.dropout(x)
        # (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        return self.linear_2(x) # W2 and B2