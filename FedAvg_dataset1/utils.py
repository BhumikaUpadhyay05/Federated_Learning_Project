import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------
# Preprocessing
# ----------------------------
def fit_save_preprocessor(df: pd.DataFrame, categorical_cols, numeric_cols, filename="preproc.pkl"):
    """
    Fit a preprocessor on the dataframe and save it.
    
    Args:
        df: Input dataframe
        categorical_cols: List of categorical column names
        numeric_cols: List of numeric column names
        filename: Path to save the preprocessor
    
    Returns:
        Fitted ColumnTransformer
    """
    ct = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ("scale", StandardScaler(), numeric_cols),
    ])
    X = df[categorical_cols + numeric_cols]
    ct.fit(X)
    joblib.dump(ct, filename)
    return ct

def load_preprocessor(filename="preproc.pkl"):
    """Load a saved preprocessor."""
    return joblib.load(filename)

def preprocess_df(df: pd.DataFrame, ct):
    """
    Preprocess dataframe using fitted ColumnTransformer.
    
    Args:
        df: Input dataframe
        ct: Fitted ColumnTransformer
    
    Returns:
        Tuple of (X_transformed, y) as numpy arrays
    """
    X = df[ct.feature_names_in_]
    Xt = ct.transform(X)
    y = df["output"].astype(int).to_numpy()
    return Xt.astype(np.float32), y.astype(np.int64)

# ----------------------------
# PyTorch Dataloaders
# ----------------------------
def create_torch_dataloader(X, y, batch_size=32, shuffle=True):
    """
    Create PyTorch DataLoader from numpy arrays.
    
    Args:
        X: Feature array
        y: Label array
        batch_size: Batch size for DataLoader
        shuffle: Whether to shuffle data
    
    Returns:
        PyTorch DataLoader
    """
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def split_train_val(X, y, val_size=0.2, random_state=42):
    """
    Split data into train and validation sets with stratification if possible.
    Falls back to non-stratified split for small datasets.
    
    Args:
        X: Feature array
        y: Label array
        val_size: Fraction of data for validation
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (X_train, X_val, y_train, y_val)
    """
    # Check if we can stratify (need at least 2 samples per class)
    unique, counts = np.unique(y, return_counts=True)
    min_class_count = counts.min()
    
    # Also check if we have enough samples to split
    total_samples = len(y)
    min_val_samples = int(total_samples * val_size)
    
    try:
        # Try stratified split if we have at least 2 samples per class
        if min_class_count >= 2 and total_samples >= 10:
            return train_test_split(X, y, test_size=val_size, random_state=random_state, stratify=y)
        else:
            # Fall back to regular split without stratification
            return train_test_split(X, y, test_size=val_size, random_state=random_state)
    except ValueError:
        # If stratification fails for any reason, do regular split
        return train_test_split(X, y, test_size=val_size, random_state=random_state)