from modules.feed_forward import FeedForward
from modules.multihead_attention import MultiHeadAttention
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob):
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, num_attention_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob)

    def forward(self, x):
        hidden_state = self.layer_norm_1(x)
        x = x + self.attention(hidden_state)
        hidden_state = self.layer_norm_2(x)
        x = x + self.feed_forward(hidden_state)
        return x
