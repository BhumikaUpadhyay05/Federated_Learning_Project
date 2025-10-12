# create_shards.py
import pandas as pd
import argparse
from utils import fit_save_preprocessor
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to the dataset csv")
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--out-dir", default="shards")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    # list categorical and numeric columns from your schema
    # REMEMBER: adjust lists if your dataset different
    categorical_cols = [
        "school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian",
        "schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"
    ]
    numeric_cols = [
        "age","Medu","Fedu","traveltime","studytime","failures","famrel","freetime",
        "goout","Dalc","Walc","health","absences","G1","G2","G3"
    ]

    # Ensure the columns exist
    cols = categorical_cols + numeric_cols + ["output"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in CSV. Check names/casing.")

    os.makedirs(args.out_dir, exist_ok=True)

    # Fit and save preprocessor on full dataset (so all clients use same mapping)
    ct = fit_save_preprocessor(df, categorical_cols, numeric_cols, filename="preproc.pkl")
    print("Saved preprocessor -> preproc.pkl")

    # shuffle and stratified-ish split by label
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Simple round-robin shard assignment to keep label distribution similar
    num_shards = args.num_shards
    shards = [ [] for _ in range(num_shards) ]
    for i, row in df.iterrows():
        shards[i % num_shards].append(row)

    for i, shard_rows in enumerate(shards):
        shard_df = pd.DataFrame(shard_rows)
        path = f"{args.out_dir}/shard_{i}.csv"
        shard_df.to_csv(path, index=False)
        print(f"Wrote shard {i} -> {path} (rows: {len(shard_df)})")

if __name__ == "__main__":
    main()
