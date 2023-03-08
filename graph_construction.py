from torch.utils.data import random_split
import random
from pathlib import Path
import tarfile
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.data import DataLoader


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.graphs = []

        # Iterate through each .tar.gz file
        for i in range(0, 10):
            file_path = self.root_dir / f"batch_1_{i}.tar.gz"

            # Extract all .pt files from the .tar.gz file
            with tarfile.open(file_path, "r:gz") as tar:
                for member in tar.getmembers():
                    if member.name.endswith(".pt"):
                        f = tar.extractfile(member)
                        self.graphs.append(torch.load(f))

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph

    def display_graph_attributes(graph):
        print(f"Number of nodes: {graph.num_nodes}")
        print(f"Number of edges: {graph.num_edges}")
        print(f"Edge index: {graph.edge_index}")
        print(f"Edge attributes: {graph.edge_attr}")
        print(f"Node attributes: {graph.x}")



