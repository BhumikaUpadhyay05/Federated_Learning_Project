# server.py
import flwr as fl
import argparse
from model import MLP
import torch
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-rounds", type=int, default=5)
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--address", type=str, default="127.0.0.1:8080")
    args = parser.parse_args()

    # Simple strategy: FedAvg (default in flwr)
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,  # fraction of clients used for training each round
        fraction_eval=1.0,
        min_fit_clients=args.num_clients,
        min_eval_clients=args.num_clients,
        min_available_clients=args.num_clients,
        eval_fn=None,  # we rely on client evaluation
    )

    print(f"Starting Flower server (address={args.address}) for {args.num_rounds} rounds")
    fl.server.start_server(server_address=args.address, config={"num_rounds": args.num_rounds}, strategy=strategy)

if __name__ == "__main__":
    main()
