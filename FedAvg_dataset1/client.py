# client.py
import argparse
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from model import MLP
from utils import load_preprocessor, preprocess_df, create_torch_dataloader, split_train_val
import pandas as pd
import numpy as np

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def get_parameters(self):
        # return parameters as list of numpy arrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=float(config.get("lr", 1e-3)))
        epochs = int(config.get("local_epochs", 1))
        for epoch in range(epochs):
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                out = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct = 0
        total = 0
        loss_total = 0.0
        with torch.no_grad():
            for Xb, yb in self.val_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                out = self.model(Xb)
                loss = self.criterion(out, yb)
                loss_total += loss.item() * Xb.size(0)
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += Xb.size(0)
        loss_avg = loss_total / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        # Flower expects (loss, num_examples, {"metric": value})
        return float(loss_avg), total, {"accuracy": float(accuracy)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="client id")
    parser.add_argument("--shard", required=True, help="path to shard csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load shard and preprocessor
    df = pd.read_csv(args.shard)
    preproc = load_preprocessor("preproc.pkl")
    X, y = preprocess_df(df, preproc)
    X_train, X_val, y_train, y_val = split_train_val(X, y, val_size=0.2)
    train_loader = create_torch_dataloader(X_train, y_train, batch_size=args.batch_size, shuffle=True)
    val_loader = create_torch_dataloader(X_val, y_val, batch_size=args.batch_size, shuffle=False)

    input_dim = X.shape[1]
    model = MLP(input_dim=input_dim, hidden_dim=64, num_classes=3).to(device)

    # Wrap in Flower client
    client = FlowerClient(model, train_loader, val_loader, device)

    # start client
    fl.client.start_numpy_client(server_address=args.server_address, client=client, config={"local_epochs": args.local_epochs, "lr": 1e-3})

if __name__ == "__main__":
    main()
