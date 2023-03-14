import torch.nn as nn
import torch


class Embeddings(nn.Module):
    def __init__(self, embed_dim, patch_size, stride, num_embeddings):
        super().__init__()
        self.projection = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=stride)
        self.L = num_embeddings
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Embedding(num_embeddings, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout()

    def forward(self, x):
        img_embeddings = self.projection(x)
        N, D, H, W = img_embeddings.shape

        position_ids = torch.arange(self.L, dtype=torch.long, device=x.device).unsqueeze(0)
        position_embeddings = self.position_embedding(position_ids)

        img_embeddings = img_embeddings.view(N, D, H * W).permute(0, 2, 1)  # [N, H*W, D]
        batch_class_token = self.class_token.expand(x.shape[0], -1, -1)
        img_embeddings = torch.cat([batch_class_token, img_embeddings], dim=1)
        embedding = img_embeddings + position_embeddings
        embedding = self.layer_norm(embedding)
        return self.dropout(embedding)
