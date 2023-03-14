import torch.nn as nn

from helpers import _scaled_dot_product_attention


class AttentionHead(nn.Module):

    def __init__(self, embed_dim, head_dim):
        """

        :param embed_dim: Token embedding dimension. Usually, [embed_dim] is chosen as a multiple of [head_dim]
        :param head_dim: Dimension to project the token vector onto
        """
        super().__init__()
        self.Q = nn.Linear(embed_dim, head_dim)
        self.K = nn.Linear(embed_dim, head_dim)
        self.V = nn.Linear(embed_dim, head_dim)

    def forward(self, hidden_state):
        """

        :param hidden_state:
        :return:
        """
        attn_out, attn_scores = _scaled_dot_product_attention(self.Q(hidden_state), self.K(hidden_state),
                                                              self.V(hidden_state))
        return attn_out, attn_scores
