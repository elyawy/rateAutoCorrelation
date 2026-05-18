"""
Add cons_decay_A and cons_decay_rho to existing features.csv
by fitting exponential decay on already-computed cons_lag*_phi columns.
No resimulation needed.
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import config

LAG_COLS = ['cons_lag1_phi', 'cons_lag2_phi', 'cons_lag5_phi', 'cons_lag10_phi', 'cons_lag20_phi']
LAGS = np.array([1, 2, 5, 10, 20], dtype=float)


def exp_decay(lag, A, decay):
    return A * np.power(decay, lag)


def fit_decay(phi_values):
    try:
        if np.any(phi_values > 1e-6):
            popt, _ = curve_fit(
                exp_decay, LAGS, phi_values,
                p0=[phi_values[0], 0.5],
                bounds=([0, 0], [1, 1]),
                maxfev=1000
            )
            return float(popt[0]), float(popt[1])
    except Exception:
        pass
    return 0.0, 0.0


def main():
    features_file = config.FEATURES_DIR / 'features.csv'
    print(f"Loading {features_file}...")
    df = pd.read_csv(features_file)
    print(f"Rows: {len(df)}")

    phi = df[LAG_COLS].values  # (N, 5)

    decay_A = np.zeros(len(df))
    decay_rho = np.zeros(len(df))

    for i, row in enumerate(phi):
        decay_A[i], decay_rho[i] = fit_decay(row)
        if i % 10000 == 0:
            print(f"  {i}/{len(df)}")

    df['cons_decay_A'] = decay_A
    df['cons_decay_rho'] = decay_rho

    df.to_csv(features_file, index=False)
    print(f"Done. Saved to {features_file}")
    print(f"  cons_decay_rho mean: {decay_rho.mean():.3f}, std: {decay_rho.std():.3f}")


if __name__ == "__main__":
    main()
