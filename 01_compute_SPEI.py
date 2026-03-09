"""
Compute SPEI (Standardised Precipitation-Evapotranspiration Index)
================================================================================
Gold standard implementation following Vicente-Serrano et al. (2010) and
Begueria et al. (2014).

Methodology:
    - Water balance: P - PET (monthly precipitation minus potential ET)
    - Rolling accumulation at 3- and 6-month timescales
    - Three-parameter Generalized Logistic (GLO) distribution
    - L-moments for parameter estimation (Hosking, 1990)
    - CDF transformation to standard normal

Inputs (relative to parent directory):
    - basin_precipitation.csv   (monthly basin-averaged precipitation)
    - basin_pet.csv             (monthly basin-averaged PET)
    - basin_inventory.csv       (basin_id <-> GAGE_ID mapping)

Output:
    - basin_drought_analysis/basin_spei.csv
      Columns: basin_id, GAGE_ID, date, timescale, SPEI  (LONG format)

Usage:
    cd Code/
    python3 01_compute_SPEI.py
================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = SCRIPT_DIR.parent          # Hydrological_Drought_Propagation/

PRECIP_FILE   = BASE_DIR / "basin_precipitation.csv"
PET_FILE      = BASE_DIR / "basin_pet.csv"
INVENTORY_FILE = BASE_DIR / "basin_inventory.csv"

OUTPUT_DIR = BASE_DIR / "basin_drought_analysis"
OUTPUT_FILE = OUTPUT_DIR / "basin_spei.csv"

TIMESCALES = [3, 6]   # months

print("=" * 80)
print("COMPUTE SPEI — GOLD STANDARD")
print("Method: L-moments + Generalized Logistic (GLO)")
print(f"Timescales: {TIMESCALES}")
print("=" * 80)

# ============================================================================
# L-MOMENTS (Hosking, 1990)
# ============================================================================

def calculate_lmoments(data):
    """
    L-moments (L1, L2, L3) via unbiased probability weighted moments.
    """
    x = np.sort(data)
    n = len(x)
    if n < 3:
        return np.nan, np.nan, np.nan

    j = np.arange(n)
    b0 = np.mean(x)
    b1 = np.sum(x * j) / (n * (n - 1))
    b2 = np.sum(x * j * (j - 1)) / (n * (n - 1) * (n - 2))

    L1 = b0
    L2 = 2 * b1 - b0
    L3 = 6 * b2 - 6 * b1 + b0
    return L1, L2, L3


# ============================================================================
# GLO DISTRIBUTION (Vicente-Serrano et al., 2010)
# ============================================================================

def fit_glo_lmoments(data):
    """
    Fit three-parameter Generalized Logistic distribution via L-moments.

    Returns (xi, alpha, kappa)  — location, scale, shape.
    """
    L1, L2, L3 = calculate_lmoments(data)
    if np.isnan(L1) or np.isnan(L2) or np.isnan(L3) or L2 == 0:
        return np.nan, np.nan, np.nan

    tau3 = np.clip(L3 / L2, -0.99, 0.99)   # L-skewness
    kappa = -tau3

    if abs(kappa) < 1e-6:
        return L1, L2, 0.0

    try:
        alpha = L2 * abs(kappa) * np.pi / np.sin(abs(kappa) * np.pi)
        xi = L1 - alpha * (1.0 / kappa - np.pi / np.sin(kappa * np.pi))
    except (ZeroDivisionError, FloatingPointError):
        return L1, L2, 0.0

    if alpha <= 0:
        alpha = abs(L2) if L2 != 0 else 1e-6

    return xi, alpha, kappa


def glo_cdf(x, xi, alpha, kappa):
    """
    CDF of the Generalized Logistic distribution.
    """
    x = np.asarray(x, dtype=float)
    if abs(kappa) < 1e-6:
        y = (x - xi) / alpha
        F = 1.0 / (1.0 + np.exp(-y))
    else:
        y = (x - xi) / alpha
        term = np.clip(1.0 - kappa * y, 1e-10, None)
        F = 1.0 / (1.0 + np.power(term, 1.0 / kappa))

    return np.clip(F, 1e-6, 1.0 - 1e-6)


# ============================================================================
# SPEI CALCULATION
# ============================================================================

def calculate_spei(water_balance, timescale):
    """
    SPEI via accumulated water balance -> GLO fit -> standard normal.
    """
    rolling_wb = water_balance.rolling(
        window=timescale, min_periods=timescale
    ).sum()

    valid = rolling_wb.dropna().values
    if len(valid) < 30:
        return pd.Series(np.nan, index=water_balance.index)

    xi, alpha, kappa = fit_glo_lmoments(valid)
    if np.isnan(xi) or np.isnan(alpha):
        return pd.Series(np.nan, index=water_balance.index)

    F = glo_cdf(rolling_wb.values, xi, alpha, kappa)
    spei = stats.norm.ppf(F)
    spei = np.clip(spei, -4.0, 4.0)
    return pd.Series(spei, index=water_balance.index)


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = datetime.now()

    # --- 1. Load inputs ------------------------------------------------
    print("\n1. Loading data...")
    precip = pd.read_csv(PRECIP_FILE)
    pet    = pd.read_csv(PET_FILE)
    inv    = pd.read_csv(INVENTORY_FILE)

    print(f"   Basins in inventory: {len(inv)}")
    print(f"   Precip records:     {len(precip):,}")
    print(f"   PET records:        {len(pet):,}")

    # Build basin_id -> GAGE_ID map
    basin_to_gage = dict(zip(inv['basin_id'], inv['GAGE_ID'].astype(str).str.zfill(8)))

    # --- 2. Prepare basin monthly data ---------------------------------
    print("\n2. Preparing monthly water balance per basin...")
    precip['date'] = pd.to_datetime(precip['date'])
    pet['date']    = pd.to_datetime(pet['date'])

    # Merge precip + PET on (basin_id, date)
    merged = precip.merge(pet, on=['basin_id', 'date'], suffixes=('_p', '_pet'))

    # Identify value columns
    p_col   = [c for c in merged.columns if c.startswith('precip') or c == 'value_p'][0]
    pet_col = [c for c in merged.columns if c.startswith('pet') or c == 'value_pet'][0]

    merged['water_balance'] = merged[p_col] - merged[pet_col]

    # --- 3. Calculate SPEI per basin -----------------------------------
    print("\n3. Calculating SPEI...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    spei_records = []
    basins = merged['basin_id'].unique()
    total = len(basins)

    for idx, basin_id in enumerate(basins):
        if (idx + 1) % 500 == 0 or idx == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (total - idx - 1) / rate if rate > 0 else 0
            print(f"   [{idx+1}/{total}] ETA {eta:.0f}s")

        bdf = merged[merged['basin_id'] == basin_id].sort_values('date').set_index('date')
        wb  = bdf['water_balance']

        if len(wb) < 30:
            continue

        gage_id = basin_to_gage.get(basin_id, '')

        for ts in TIMESCALES:
            spei = calculate_spei(wb, ts)
            valid = spei.dropna()
            for date, val in valid.items():
                spei_records.append({
                    'basin_id':  basin_id,
                    'GAGE_ID':   gage_id,
                    'date':      date.strftime('%Y-%m-%d'),
                    'timescale': ts,
                    'SPEI':      round(val, 4)
                })

    # --- 4. Save -------------------------------------------------------
    print("\n4. Saving output...")
    df = pd.DataFrame(spei_records)
    df.to_csv(OUTPUT_FILE, index=False)

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*80}")
    print("SPEI COMPUTATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Records:   {len(df):,}")
    print(f"  Basins:    {df['basin_id'].nunique():,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Output:    {OUTPUT_FILE}")
    print(f"  Time:      {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
