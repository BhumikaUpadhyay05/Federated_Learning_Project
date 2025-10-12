# utils.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader

def fit_save_preprocessor(df: pd.DataFrame, categorical_cols, numeric_cols, filename="preproc.pkl"):
    ct = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols),
        ("scale", StandardScaler(), numeric_cols),
    ])
    X = df[categorical_cols + numeric_cols]
    ct.fit(X)
    joblib.dump(ct, filename)
    return ct

def load_preprocessor(filename="preproc.pkl"):
    return joblib.load(filename)

def preprocess_df(df: pd.DataFrame, ct):
    # ct expects the columns in the same order as fit
    X = df[ct.feature_names_in_]
    Xt = ct.transform(X)
    y = df["output"].astype(int).to_numpy()
    return Xt.astype(np.float32), y.astype(np.int64)

def create_torch_dataloader(X, y, batch_size=32, shuffle=True):
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def split_train_val(X, y, val_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)
