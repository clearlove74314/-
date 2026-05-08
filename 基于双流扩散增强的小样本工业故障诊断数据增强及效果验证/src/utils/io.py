import os
import numpy as np
import pandas as pd


def save_np_array(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def load_np_array(path):
    return np.load(path)


def save_csv(path, df):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_csv(path):
    return pd.read_csv(path)


def save_dict(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for key, value in data.items():
            f.write(f'{key}: {value}\n')