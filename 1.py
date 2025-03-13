import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, SAGEConv

class PHGTLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super(PHGTLayer, self).__init__()
        # Multi-head attention captures interactions among tokens
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)

        # Feed-forward network enhances token representations
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [sequence_length, batch_size, hidden_dim]

        # Multi-head attention
        attn_output, _ = self.mha(x, x, x)
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm1(x + ffn_output)
        
        return x

class PHGT(nn.Module):
    def __init__(self, hidden_dim, num_classes, metadata, num_heads=4):
        super(PHGT, self).__init__()
        self.hidden_dim = hidden_dim

        # Structure encoder using heterogeneous graph convolution
        self.structure_encoder = HeteroConv({
            edge_type: SAGEConv((-1, -1), hidden_dim)
            for edge_type in metadata[1]
        })

        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            PHGTLayer(hidden_dim, num_heads) for _ in range(2)
        ])

        # Classifier head for node classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def semantic_token(self, meta_path_instances, node_emb):
        # Aggregate node embeddings along meta-paths to form semantic tokens
        semantic_tokens = []
        for path in meta_path_instances:
            path_emb = node_emb[path].mean(dim=0)
            semantic_tokens.append(path_emb)
        semantic_tokens = torch.stack(semantic_tokens)
        return semantic_tokens.unsqueeze(1)

    def global_token(self, clusters, node_emb):
        # Aggregate node embeddings within clusters to form global tokens
        global_tokens = []
        for cluster in clusters:
            global_emb = torch.mean(node_emb[cluster], dim=0)
            global_tokens.append(global_emb)
        return torch.stack(global_tokens).unsqueeze(1)

    def forward(self, data, meta_path_instances, clusters, target_node_type):
        # Structure encoding step
        node_emb = self.structure_encoder(data.x_dict, data.edge_index_dict)[target_node_type]

        # Prepare node tokens
        node_tokens = node_emb.unsqueeze(0)

        # Generate semantic tokens
        semantic_tokens = self.semantic_token(meta_path_instances, node_emb)

        # Generate global tokens
        global_tokens = self.global_token(clusters, node_emb).unsqueeze(0)

        # Concatenate node, semantic, and global tokens
        tokens = torch.cat([node_tokens, semantic_tokens, global_tokens], dim=0)

        # Apply transformer layers
        for layer in self.transformer_layers:
            tokens = layer(tokens)

        # Extract node representation (first token) and classify
        final_repr = tokens[0]
        logits = self.classifier(final_repr)

        return logits
