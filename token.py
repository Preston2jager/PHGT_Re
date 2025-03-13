def semantic_token(meta_path_instances, node_emb):
    tokens = []
    for path in meta_path_instances:
        path_emb = torch.mean(node_emb[path], dim=0)
        tokens.append(path_emb)
    return torch.stack(tokens)  # [num_semantic_tokens, emb_dim]

def global_token(clusters, node_emb):
    global_tokens = []
    for cluster in clusters:
        global_emb = torch.mean(node_emb[cluster], dim=0)
        global_tokens.append(global_emb)
    return torch.stack(global_tokens)  # [num_global_tokens, dim]
