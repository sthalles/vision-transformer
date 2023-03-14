from modules.embeddings import Embeddings
from modules.encoder_layer import TransformerEncoderLayer
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, *, patch_size, stride, num_embeddings, hidden_size, num_hidden_layers,
                 num_attention_heads, intermediate_size, hidden_dropout_prob):
        super(TransformerEncoder, self).__init__()
        self.embeddings = Embeddings(hidden_size, patch_size, stride, num_embeddings)
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(hidden_size, num_attention_heads, intermediate_size, hidden_dropout_prob) for
            _ in range(num_hidden_layers))

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x)
        return x
