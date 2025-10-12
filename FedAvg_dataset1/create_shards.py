import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib


dataset_path = "dataset1_with_output.csv"  #  dataset
num_shards = 20
out_dir = "shards"
preproc_file = "preproc.pkl"


def create_shards(csv_path, num_shards=20, out_dir="shards", preproc_file="preproc.pkl"):
    # Load dataset
    df = pd.read_csv(csv_path)
    if "output" not in df.columns:
        raise ValueError("Column 'output' not found in dataset!")

    # Define categorical and numeric columns
    categorical_cols = [
        "school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian",
        "schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"
    ]
    numeric_cols = [
        "age","Medu","Fedu","traveltime","studytime","failures","famrel","freetime",
        "goout","Dalc","Walc","health","absences","G1","G2","G3"
    ]

    # Fit preprocessor
    ct = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("scale", StandardScaler(), numeric_cols),
    ])
    ct.fit(df[categorical_cols + numeric_cols])
    joblib.dump(ct, preproc_file)
    print(f"Saved preprocessor -> {preproc_file}")

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into shards
    shards = []
    shard_size = len(df) // num_shards
    for i in range(num_shards):
        start = i * shard_size
        end = (i + 1) * shard_size if i != num_shards - 1 else len(df)
        shard_df = df.iloc[start:end]
        shards.append(shard_df)

    # Create output folder
    os.makedirs(out_dir, exist_ok=True)

    # Save shards
    for idx, shard_df in enumerate(shards):
        path = os.path.join(out_dir, f"shard_{idx}.csv")
        shard_df.to_csv(path, index=False)
        print(f"Shard {idx} saved -> {path} (rows: {len(shard_df)})")

if __name__ == "__main__":
    create_shards(dataset_path, num_shards, out_dir, preproc_file)
