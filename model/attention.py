import torch
import torch.nn as nn
from torchtyping import TensorType

class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False) # B = Batch, T = custom sequence length based on embedding, attention_dim
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        pass

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        k = self.key(embedded)       # (B, T, attention_dim)
        q = self.query(embedded)     # (B, T, attention_dim)
        v = self.value(embedded)     # (B, T, attention_dim)

        # attention score
        raw_score = q @ k.transpose(-2, -1) # we use -2 and -1 because we ALWAYS want to swap the 2nd last and last row.
        context_length, attention_dim = k.shape[1], k.shape[2] # context length is just T or the embedding_dim
        score = raw_score / (attention_dim ** 0.5)
        
        #causal mask: prevent attending to future tokens
        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        score = score.masked_fill(mask, float('-inf'))
        score = nn.functional.softmax(score, dim=2)

        return torch.round(score @ v, decimals=4)

        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        pass
