import torch.nn as nn
import torch

from modules.attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_attention_heads):
        super().__init__()
        head_dim = embed_dim // num_attention_heads
        self.heads = nn.ModuleList([AttentionHead(embed_dim, head_dim) for _ in range(num_attention_heads)])
        self.linear_projection = nn.Linear(embed_dim, embed_dim)

    def forward(self, hidden_state):
        attn_out = []
        attn_weights = []
        for head in self.heads:
            out, weights = head(hidden_state)
            attn_out.append(out)
            attn_weights.append(weights)

        attn_out = torch.cat(attn_out, dim=-1)
        attn_weights = torch.stack(attn_weights, dim=1)
        out = self.linear_projection(attn_out)

        return out
