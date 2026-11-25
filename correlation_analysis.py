"""
Compute the mean absolute Pearson correlation (rho_bar) of each benchmark dataset.

This is a purely data-driven measure: it depends only on the dataset itself, not on
any model. The model-side gain (how much CMIE improves over CALF) is handled
separately in correlation_gain.py.

Usage:
    python correlation_analysis.py [--data_root ./datasets/]

The printed rho_bar values go into Table 1 (channel correlation column).
"""
import argparse
import os

import numpy as np
import pandas as pd

# dataset -> relative csv path (under --data_root)
DATASET_FILES = {
    'ETTh1': 'ETT-small/ETTh1.csv',
    'ETTh2': 'ETT-small/ETTh2.csv',
    'ETTm1': 'ETT-small/ETTm1.csv',
    'ETTm2': 'ETT-small/ETTm2.csv',
    'Weather': 'weather/weather.csv',
    'Electricity': 'electricity/electricity.csv',
    'Traffic': 'traffic/traffic.csv',
}

# thresholds mapping rho_bar -> weak / moderate / strong
WEAK, STRONG = 0.3, 0.5


def label_of(rho):
    if rho < WEAK:
        return 'weak'
    if rho < STRONG:
        return 'moderate'
    return 'strong'


def mean_abs_pearson(X):
    """X: (T, C) numpy array. Mean absolute Pearson correlation over all channel pairs."""
    C = X.shape[1]
    if C < 2:
        return 0.0
    corr = np.corrcoef(X.T)  # (C, C)
    corr = np.abs(np.nan_to_num(corr, nan=0.0))
    iu = np.triu_indices(C, k=1)  # upper triangle, excluding diagonal
    return float(corr[iu].mean())


def compute_correlations(data_root):
    """Return a list of (name, num_channels, rho_bar, label) for every dataset."""
    rows = []
    for name, rel in DATASET_FILES.items():
        path = os.path.join(data_root, rel)
        df = pd.read_csv(path)
        cols = [c for c in df.columns if c != 'date']
        X = df[cols].values.astype(float)  # (T, C)
        rho = mean_abs_pearson(X)
        rows.append((name, X.shape[1], rho, label_of(rho)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/')
    args = parser.parse_args()

    rows = compute_correlations(args.data_root)

    print('\n=== Summary (dataset | C | rho_bar | label) ===')
    for name, C, rho, lab in rows:
        print(f"{name:12s} {C:4d} {rho:.4f} {lab:8s}")


if __name__ == '__main__':
    main()