import torch.nn as nn

class PHGTBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim*4), nn.ReLU(), nn.Linear(dim*4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        h = self.norm1(x + self.mha(x, x, x)[0])
        h = self.norm2(h + self.ffn(h))
        return h

token_seq = torch.cat([node_tokens, semantic_tokens, global_tokens], dim=0)  # [T, dim]


transformer_layers = nn.ModuleList([PHGTTransformerLayer(dim) for _ in range(L)])

for layer in transformer_layers:
    poly_tokens = layer(poly_tokens)

class NodeClassifier(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Dropout(0.5), nn.Linear(dim, num_classes))

    def forward(self, node_embeddings):
        return self.classifier(node_embeddings)

