optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    poly_tokens = model.encode_structure(data)
    node_tokens = poly_tokens[data.target_nodes]

    semantic_tokens = semantic_token(meta_path_instances, poly_tokens)
    global_tokens = global_token(clusters, poly_tokens)
    tokens = torch.cat([node_tokens, semantic_tokens, global_tokens], dim=0)
    
    logits = model.transformer(tokens)
    pred = classifier(logits)
    
    loss = criterion(pred[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
