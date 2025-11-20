# -----------------------
# EVT pipeline with AIC and Q-Q plots
# -----------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme, expon
import math

EXAMPLE_IMAGE_PATH = "/mnt/data/7db574f7-61a0-4421-8ed5-1cac64a30224.jpg"

# -----------------------
# Generate synthetic tornado dataset 1925–2024
# -----------------------
np.random.seed(42)
years = np.arange(1925, 2025)
rows = []

ef_probs = [0.3, 0.4, 0.15, 0.1, 0.04, 0.01]  # EF0–EF5 probabilities
ef_to_mph = {0:(65,85), 1:(86,110), 2:(111,135), 3:(136,165), 4:(166,200), 5:(201,300)}

for year in years:
    n_tornadoes = np.random.randint(5, 16)
    for _ in range(n_tornadoes):
        ef = np.random.choice([0,1,2,3,4,5], p=ef_probs)
        wind = np.mean(ef_to_mph[ef])
        month = np.random.randint(1, 13)
        day = np.random.randint(1, 29)
        date = pd.Timestamp(year=year, month=month, day=day)
        rows.append({'Date': date, 'EF': ef, 'Wind_mph': wind})

df = pd.DataFrame(rows).sort_values('Date').reset_index(drop=True)

# Tri-State tornado outlier
tri_state_date = pd.Timestamp('1925-03-18')
if (df['Date'] == tri_state_date).any():
    df.loc[df['Date'] == tri_state_date, 'Wind_mph'] = 275
    df.loc[df['Date'] == tri_state_date, 'EF'] = 5
else:
    df = pd.concat([pd.DataFrame([{'Date': tri_state_date, 'EF':5, 'Wind_mph':275}]), df], ignore_index=True)
    df = df.sort_values('Date').reset_index(drop=True)

# -----------------------
# EVT pipeline
# -----------------------
def run_evt_pipeline(df, return_periods=(50,100), threshold_quantile=0.95):
    results = {}
    wind = df['Wind_mph'].dropna()

    # Block maxima: annual maxima
    df['Year'] = df['Date'].dt.year
    annual_max = df.groupby('Year')['Wind_mph'].max()
    results['annual_maxima_series'] = annual_max

    # Fit GEV to annual maxima
    c, loc, scale = genextreme.fit(annual_max)
    results['GEV_params'] = {'c': c, 'loc': loc, 'scale': scale}

    # GEV return levels
    return_levels = {rp: genextreme.ppf(1 - 1.0/rp, c, loc=loc, scale=scale) for rp in return_periods}
    results['GEV_return_levels'] = return_levels

    # POT threshold
    thresh = wind.quantile(threshold_quantile)
    exceed_mask = wind > thresh
    excesses = (wind[exceed_mask] - thresh).reset_index(drop=True)
    results['threshold'] = thresh
    results['excesses'] = excesses
    results['exceedance_indices'] = wind[exceed_mask].index.to_numpy()
    results['exceedance_values'] = wind[exceed_mask].to_numpy()

    # Exponential fit to excesses
    if len(excesses) > 0:
        loc_pot, scale_pot = expon.fit(excesses)
    else:
        loc_pot, scale_pot = 0.0, np.nan
    results['POT_params'] = {'loc': loc_pot, 'scale': scale_pot}

    # -----------------------
    # Model evaluation: AIC
    # -----------------------
    def compute_aic_gev(params, data):
        c, loc, scale = params
        ll = np.sum(genextreme.logpdf(data, c, loc=loc, scale=scale))
        k = 3
        return 2*k - 2*ll

    def compute_aic_exp(params, data):
        loc, scale = params
        ll = np.sum(expon.logpdf(data, loc=loc, scale=scale))
        k = 1
        return 2*k - 2*ll

    results['GEV_AIC'] = compute_aic_gev(list(results['GEV_params'].values()), annual_max)
    results['POT_AIC'] = compute_aic_exp([loc_pot, scale_pot], excesses) if len(excesses) > 0 else np.nan

    # -----------------------
    # Diagnostic Q-Q plots
    # -----------------------
    # GEV Q-Q
    plt.figure(figsize=(6,6))
    sorted_data = np.sort(annual_max)
    n = len(sorted_data)
    prob = (np.arange(1, n+1) - 0.5) / n
    theoretical_q = genextreme.ppf(prob, c, loc=loc, scale=scale)
    plt.scatter(theoretical_q, sorted_data, color='steelblue')
    plt.plot([min(sorted_data), max(sorted_data)], [min(sorted_data), max(sorted_data)], 'r--', lw=2)
    plt.xlabel("GEV Theoretical Quantiles")
    plt.ylabel("Observed Annual Maxima")
    plt.title("Q-Q Plot: GEV Fit")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("qqplot_gev.png")
    plt.show()

    # POT Exponential Q-Q
    if len(excesses) > 0:
        plt.figure(figsize=(6,6))
        sorted_excesses = np.sort(excesses)
        n = len(sorted_excesses)
        prob = (np.arange(1, n+1) - 0.5) / n
        theoretical_q = expon.ppf(prob, loc=loc_pot, scale=scale_pot)
        plt.scatter(theoretical_q, sorted_excesses, color='darkorange')
        plt.plot([min(sorted_excesses), max(sorted_excesses)], [min(sorted_excesses), max(sorted_excesses)], 'r--', lw=2)
        plt.xlabel("Exponential Theoretical Quantiles")
        plt.ylabel("Observed Excesses")
        plt.title("Q-Q Plot: POT Exponential Fit")
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig("qqplot_pot.png")
        plt.show()

    return results

results = run_evt_pipeline(df, return_periods=(50,100), threshold_quantile=0.95)

# -----------------------
# Print summary
# -----------------------
print("GEV params:", results['GEV_params'])
print(f"GEV AIC: {results['GEV_AIC']:.2f}")
print("GEV return levels:", results['GEV_return_levels'])
print("POT params:", results['POT_params'])
print(f"POT AIC: {results['POT_AIC']:.2f}" if not np.isnan(results['POT_AIC']) else "POT AIC: N/A")
print("POT threshold (95%):", results['threshold'])
print(f"Number of exceedances: {len(results['excesses'])}")

# -----------------------
# Existing plots (annual maxima, POT bars, block maxima, POT histogram) can remain unchanged
# -----------------------
