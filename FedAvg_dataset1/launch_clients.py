import subprocess
import time

num_clients = 20
local_epochs = 5  # number of local epochs per client

for i in range(num_clients):
    subprocess.Popen([
        "python", "client.py",
        f"--cid={i}",
        f"--shard=shards/shard_{i}.csv",
        f"--local-epochs={local_epochs}"
    ])
    time.sleep(0.2)