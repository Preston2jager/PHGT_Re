from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec

meta_paths = [('author', 'paper', 'author'), ('author', 'paper', 'conference', 'paper', 'author')]
