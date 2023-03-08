from graph_construction import CustomDataset
from model import InteractionNetwork
from traning import train_model

from torch.utils.data import random_split
import random
from pathlib import Path
import tarfile
import torch
# from torch.utils.data import Dataset
# from torch_geometric.data import Data
from torch_geometric.data import DataLoader
import sys


def main():

    # Dataset created from the tar files
    print("Dataset creating from the tar files")
    dataset = CustomDataset("tar_data")
    print(f"Number of graphs: {len(dataset)}")

    # Iterate through the dataset and print the first graph
    for i in range(len(dataset)):
        print(f"Graph {i+1}:")
        print((dataset[i]))
        break

    # Shuffle dataset
    random.shuffle(dataset.graphs)

    # Split dataset into train and eval
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = random_split(dataset, [train_size, eval_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32)


    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InteractionNetwork(10).to(device)

    arg1 = sys.argv[1]
    if arg1 == "balanced":
        print("Using balanced loss function")
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        print("Using unbalanced loss function")

        # Train with class-balanced binary cross-entropy loss
        n_positives = sum([dataset[i].y.eq(1).sum().item()
                        for i in range(len(dataset))])
        n_negatives = sum([dataset[i].y.eq(0).sum().item()
                        for i in range(len(dataset))])
        pos_weight = torch.tensor(n_negatives / n_positives, dtype=torch.float32)
        bce_loss_class_balanced = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        criterion = bce_loss_class_balanced

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, eval_loader,
                bce_loss_class_balanced, optimizer, 50)



if __name__ == "__main__":
    main()