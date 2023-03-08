import torch

def train_model(model, train_loader, eval_loader, criterion, optimizer, n_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(n_epochs):
        # Train loop
        train_loss = 0
        correct = 0
        total = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr).squeeze(1)
            # print(out.shape)
            # print(data.y.shape)
            # break
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            predicted = torch.round(torch.sigmoid(out))
            correct += (predicted == data.y).sum().item()
            total += data.y.size(0)

            train_loss += loss.item()
        # break
        train_acc = 100 * correct / total
        train_loss /= len(train_loader)

        # Evaluation loop
        eval_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in eval_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr).squeeze(1)
                loss = criterion(out, data.y)

                # Calculate accuracy
                predicted = torch.round(torch.sigmoid(out))
                correct += (predicted == data.y).sum().item()
                total += data.y.size(0)

                eval_loss += loss.item()

        eval_acc = 100 * correct / total
        eval_loss /= len(eval_loader)

        # Print epoch stats
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Eval Loss: {eval_loss:.4f}, Eval Acc: {eval_acc:.2f}%")

    print("Training finished!")

