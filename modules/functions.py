import math
import torch


def _scaled_dot_product_attention(query, key, value):
    """
    Perform dot product attention between Q, K, V
    :param query: 3D tensor of shape [N, L, D]
    :param key: 3D tensor of shape [M, L, D]
    :param value: 3D tensor of shape [N, L, D]
    :return: 3D tensor of shape [N, L, D]
    """
    assert len(query.shape) == 3
    assert len(query.shape) == len(key.shape) == len(value.shape)
    dim = query.shape[-1]
    weights = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim)
    scores = torch.softmax(weights, dim=-1)
    return torch.bmm(scores, value), scores
