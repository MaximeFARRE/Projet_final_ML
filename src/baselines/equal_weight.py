# Equal-Weight Buy & Hold baseline.
#invest the same fraction of capital in each asset at the beginning
#and portfolio value follows the average performance of all assets


import numpy as np
import pandas as pd

# Compute the equity curve of a simple Equal-Weight Buy & Hold strategy.

def equity_curve_equal_weight(prices_df: pd.DataFrame) -> pd.Series:
   
    # Drop dates with missing prices
    prices_df = prices_df.dropna()

    n_assets = prices_df.shape[1]
    if n_assets == 0:
        raise ValueError("Prices DataFrame contains no assets.")

    # Equal weights: 1/N for each asset
    weights = np.full(n_assets, 1.0 / n_assets)

    # Price relatives compared to the first day
    relatives = prices_df / prices_df.iloc[0]

    # Portfolio equity curve: weighted average of relatives
    equity_curve = (relatives * weights).sum(axis=1)
    equity_curve.name = "equity_equal_weight"

    return equity_curve
