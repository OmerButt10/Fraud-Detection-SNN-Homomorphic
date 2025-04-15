import numpy as np  # type: ignore

import pandas as pd  # type: ignore

from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.preprocessing import StandardScaler  # type: ignore


# -------------------------
# Preprocessing and loading dataset function
# -------------------------
def load_dataset(path: str, frac: float, val_split: float):
    # -------------------------
    # Data Loading & Preprocessing
    # -------------------------
    df = pd.read_csv(path)

    # Drop ID columns (non-numeric or irrelevant)
    df = df.drop(columns=['nameOrig', 'nameDest'])

    # Balance the dataset: take equal samples from each class
    df_0 = df[df['isFraud'] == 0]
    df_1 = df[df['isFraud'] == 1]
    n_minority = min(len(df_0), len(df_1))
    df_balanced_0 = df_0.sample(n=n_minority, random_state=42)
    df_balanced_1 = df_1.sample(n=n_minority, random_state=42)
    df_balanced = pd.concat([df_balanced_0, df_balanced_1]).sample(frac=1, random_state=42)
    
    balanced_counts = df_balanced['isFraud'].value_counts().to_dict()
    print("Balanced dataset counts:", balanced_counts)

    # Sample a fraction of the balanced dataset
    df_sample = df_balanced.sample(frac=frac, random_state=42)

    # Separate target and features
    y = df_sample['isFraud'].values
    X = df_sample.drop(columns=['isFraud']).values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data (stratified split)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y, test_size=0.3, stratify=y, random_state=42
    )
    print("\nSplit data into train_val and test sets.")
    print("Train_val shape:", X_train_val.shape, "Test shape:", X_test.shape)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_split, stratify=y_train_val, random_state=42
    )

    return X_train, X_test, X_val, y_val, y_train, y_test

 
# -------------------------
# Convert Data to Spike Trains (Rate Coding)
# -------------------------
def rate_code(values, max_rate=100, time_steps=10):  # TODO: default 50
    """
    Convert numerical values to spike trains using rate coding.
    """
    firing_rates = (values - values.min()) / (values.max() - values.min() + 1e-8) * max_rate
    spikes = np.random.rand(time_steps, len(values)) < (firing_rates / max_rate)
    return spikes.astype(np.float32)
