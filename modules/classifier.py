from modules.encoder import TransformerEncoder
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        number_of_patches = (args.image_size // args.patch_size) ** 2
        self.encoder = TransformerEncoder(patch_size=args.patch_size, stride=args.stride,
                                          num_embeddings=number_of_patches + 1,
                                          hidden_size=args.hidden_size, num_hidden_layers=args.num_encoder_layers,
                                          num_attention_heads=args.num_attention_heads,
                                          hidden_dropout_prob=args.hidden_dropout_prob,
                                          intermediate_size=args.intermediate_size)
        self.linear = nn.Linear(args.hidden_size, args.num_classes)

    def forward(self, x):
        out = self.encoder(x)
        return self.linear(out[:, 0, :])
