"""
Relate each dataset's channel correlation (rho_bar) to CMIE's gain over CALF, and
visualize the relationship with a scatter plot.

The rho_bar values are computed by correlation_analysis.py (purely data-driven,
dataset-only). This script adds the model-side gain, which requires the model MSE
reported in Table 1 of the manuscript.

Usage:
    python correlation_gain.py [--data_root ./datasets/] [--out correlation_gain.png]
"""
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from correlation_analysis import WEAK, STRONG, label_of, compute_correlations

# Long-term forecasting MSE, averaged over pred_len in {96, 192, 336, 720},
# taken from Table 1 of the manuscript.
# TODO: update these if the long-term results table changes.
CMIE_MSE = {
    'ETTh1': 0.432, 'ETTh2': 0.347, 'ETTm1': 0.385, 'ETTm2': 0.277,
    'Weather': 0.248, 'Electricity': 0.174, 'Traffic': 0.393,
}
CALF_MSE = {
    'ETTh1': 0.431, 'ETTh2': 0.349, 'ETTm1': 0.395, 'ETTm2': 0.281,
    'Weather': 0.250, 'Electricity': 0.175, 'Traffic': 0.439,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./datasets/')
    parser.add_argument('--out', type=str, default='correlation_gain.png')
    args = parser.parse_args()

    # rho_bar values from Table 1 of the manuscript (fallback when raw data is unavailable)
    RHO_BAR = {
        'ETTh1': 0.2221, 'ETTh2': 0.3246, 'ETTm1': 0.2243, 'ETTm2': 0.3245,
        'Weather': 0.2956, 'Electricity': 0.4893, 'Traffic': 0.5638,
    }

    try:
        rows = compute_correlations(args.data_root)  # (name, C, rho, label)
        names = [r[0] for r in rows]
        rho = [r[2] for r in rows]
        lab = [r[3] for r in rows]
    except Exception:
        names = list(CMIE_MSE.keys())
        rho = [RHO_BAR[n] for n in names]
        lab = [label_of(r) for r in rho]
    gain = [(CALF_MSE[n] - CMIE_MSE[n]) / CALF_MSE[n] * 100.0 for n in names]

    print('\n=== Summary (dataset | rho_bar | label | gain%) ===')
    for n, x, l, g in zip(names, rho, lab, gain):
        print(f"{n:12s} {x:.4f} {l:8s} {g:+6.2f}")

    # scatter: correlation strength vs. CMIE gain over CALF
    plt.figure(figsize=(7, 5))
    plt.scatter(rho, gain, s=80, color='#1f77b4', zorder=3)
    for n, x, y in zip(names, rho, gain):
        plt.annotate(n, (x, y), textcoords='offset points', xytext=(6, 4), fontsize=9)
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    plt.axvline(WEAK, color='lightgray', linestyle=':', linewidth=0.8)
    plt.axvline(STRONG, color='lightgray', linestyle=':', linewidth=0.8)
    plt.xlabel(r'Mean absolute Pearson correlation $\bar{\rho}$')
    plt.ylabel('CMIE gain over CALF (%)')
    plt.title('Channel correlation strength vs. CMIE gain')
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'\nSaved scatter plot to {args.out}')


if __name__ == '__main__':
    main()