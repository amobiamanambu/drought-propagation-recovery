"""
Compute SSI (Standardised Streamflow Index) with Integrated Quality Control
================================================================================
Fuses discharge quality control, SSI computation, and persistent-aridity
filtering into a single reproducible script.

Pipeline (all executed in sequence):
    A.  Read daily discharge from USGS RDB batch files.
    B.  Quality-control each gage:
            Tier 1 — >=90 % completeness AND >=30 yr record
        Only Tier 1 gages proceed to SSI computation.
    C.  Aggregate daily discharge to monthly means.
    D.  Compute SSI at 3- and 6-month accumulations using:
            - Gamma distribution (McKee et al., 1993)
            - L-moments parameter estimation (Hosking, 1990)
            - Separate fitting for each calendar month (seasonality)
            - Mixed distribution for zero-flow handling
    E.  Flag persistently arid basins (>50 % drought frequency at
        SSI <= -1.0; Dai, 2011; Van Loon, 2015) and remove them.

Inputs (relative to parent directory):
    - basin_inventory.csv
    - usgs_dv_00060_1980_01_01_to_2025_12_31/manifest.csv
    - usgs_dv_00060_1980_01_01_to_2025_12_31/raw_rdb_gz/*.rdb.gz

Outputs:
    - basin_drought_indices/basin_SSI_timeseries.csv
      Columns: basin_id, GAGE_ID, date, SSI_3, SSI_6  (WIDE format, non-arid
      Tier 1 basins only)
    - quality_control/tier1_non_arid_basins.csv

Usage:
    cd Code/
    python3 02_compute_SSI.py
================================================================================
"""

import os
import gzip
import numpy as np
import pandas as pd
from scipy import special
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR   = SCRIPT_DIR.parent

BASIN_INVENTORY = BASE_DIR / "basin_inventory.csv"
DISCHARGE_DIR   = BASE_DIR / "usgs_dv_00060_1980_01_01_to_2025_12_31" / "raw_rdb_gz"
MANIFEST_FILE   = BASE_DIR / "usgs_dv_00060_1980_01_01_to_2025_12_31" / "manifest.csv"

SSI_OUTPUT_DIR  = BASE_DIR / "basin_drought_indices"
QC_OUTPUT_DIR   = BASE_DIR / "quality_control"
SSI_OUTPUT_FILE = SSI_OUTPUT_DIR / "basin_SSI_timeseries.csv"
TIER1_OUTPUT    = QC_OUTPUT_DIR / "tier1_non_arid_basins.csv"

TIMESCALES = [3, 6]

# Quality thresholds
COMPLETENESS_TIER1 = 90.0   # %
RECORD_LENGTH_TIER1 = 30    # years

# Aridity thresholds
DROUGHT_THRESHOLD = -1.0    # SSI <= this = drought month
ARIDITY_THRESHOLD = 50.0    # >50 % drought frequency = persistently arid

START_DATE = pd.Timestamp('1980-01-01')
END_DATE   = pd.Timestamp('2025-12-31')

print("=" * 80)
print("COMPUTE SSI — GOLD STANDARD + QC + ARIDITY FILTER")
print(f"Timescales: {TIMESCALES}")
print("=" * 80)

# ============================================================================
# RDB PARSER (robust multi-section)
# ============================================================================

def parse_rdb_batch_file(filepath, target_gages):
    """
    Parse a gzipped multi-section USGS RDB batch file.
    Returns {gage_id: DataFrame[datetime, discharge]} for target gages only.
    """
    site_data = {}
    try:
        with gzip.open(filepath, 'rt') as f:
            lines = f.readlines()
    except Exception:
        return site_data

    lines = [line for line in lines if not line.startswith('#')]
    if len(lines) < 3:
        return site_data

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('agency_cd'):
            columns = line.split('\t')
            i += 1            # skip format line
            if i >= len(lines):
                break
            i += 1            # first data row

            data_rows = []
            while i < len(lines) and not lines[i].startswith('agency_cd'):
                data_rows.append(lines[i].strip())
                i += 1

            if len(data_rows) == 0:
                continue

            try:
                parsed = [row.split('\t') for row in data_rows if len(row.split('\t')) >= 3]
                if len(parsed) == 0:
                    continue

                df = pd.DataFrame(parsed, columns=columns[:len(parsed[0])])
                if 'site_no' not in df.columns:
                    continue

                site_no = str(df['site_no'].iloc[0]).zfill(8)
                if site_no not in target_gages:
                    continue

                if 'datetime' not in df.columns:
                    continue

                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df = df[df['datetime'].notna()].copy()

                discharge_cols = [c for c in df.columns
                                  if '00060' in str(c) and not c.endswith('_cd')]
                if len(discharge_cols) == 0:
                    continue

                df['discharge'] = pd.to_numeric(df[discharge_cols[0]], errors='coerce')
                df = df[(df['datetime'] >= START_DATE) &
                        (df['datetime'] <= END_DATE)].copy()

                if len(df) > 0:
                    site_data[site_no] = df[['datetime', 'discharge']].copy()
            except Exception:
                pass
        else:
            i += 1

    return site_data


# ============================================================================
# L-MOMENTS + GAMMA DISTRIBUTION (Gold Standard)
# ============================================================================

def calculate_lmoments(data):
    """L-moments (L1, L2, L3) via unbiased PWMs (Hosking, 1990)."""
    x = np.sort(data[~np.isnan(data)])
    n = len(x)
    if n < 3:
        return None

    j = np.arange(n)
    b0 = np.mean(x)
    b1 = np.sum(x * j) / (n * (n - 1))
    b2 = np.sum(x * j * (j - 1)) / (n * (n - 1) * (n - 2))

    return {'L1': b0, 'L2': 2 * b1 - b0, 'L3': 6 * b2 - 6 * b1 + b0}


def fit_gamma_lmoments(data):
    """Fit Gamma distribution via L-moments.  Returns {alpha, beta} or None."""
    pos = data[data > 0]
    if len(pos) < 20:
        return None

    lm = calculate_lmoments(pos)
    if lm is None or lm['L1'] <= 0 or lm['L2'] <= 0:
        return None

    t = lm['L2'] / lm['L1']
    if 0 < t < 0.5:
        alpha = (1 - 0.3080 * t) / (t - 0.05812 * t**2 + 0.01765 * t**3)
        if alpha <= 0 or alpha > 1000:
            mean_v, var_v = np.mean(pos), np.var(pos, ddof=1)
            if var_v <= 0:
                return None
            alpha, beta = mean_v**2 / var_v, var_v / mean_v
        else:
            beta = lm['L1'] / alpha
    else:
        mean_v, var_v = np.mean(pos), np.var(pos, ddof=1)
        if var_v <= 0 or mean_v <= 0:
            return None
        alpha, beta = mean_v**2 / var_v, var_v / mean_v

    if alpha <= 0 or beta <= 0 or np.isnan(alpha) or np.isnan(beta):
        return None
    return {'alpha': alpha, 'beta': beta}


def gamma_cdf(x, alpha, beta):
    """CDF of the Gamma distribution via regularised incomplete gamma."""
    with np.errstate(invalid='ignore', divide='ignore'):
        return special.gammainc(alpha, x / beta)


# ============================================================================
# STANDARD NORMAL PPF (rational approximation)
# ============================================================================

def _norm_ppf_scalar(p):
    """Inverse standard normal CDF (scalar)."""
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    else:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

norm_ppf_vec = np.vectorize(_norm_ppf_scalar)


# ============================================================================
# VECTORISED SSI CALCULATION (per calendar month)
# ============================================================================

def calculate_ssi(monthly_discharge, timescale):
    """
    SSI via gamma + L-moments + per-calendar-month fitting + zero handling.
    Returns pd.Series of SSI values.
    """
    n = len(monthly_discharge)
    if n < 30:
        return pd.Series(np.nan, index=monthly_discharge.index)

    if timescale == 1:
        accumulated = monthly_discharge.values.copy()
    else:
        accumulated = pd.Series(monthly_discharge.values).rolling(
            window=timescale, min_periods=timescale
        ).sum().values

    months = monthly_discharge.index.month.values
    ssi = np.full(n, np.nan)

    for month in range(1, 13):
        mask = months == month
        indices = np.where(mask)[0]
        mdata = accumulated[mask]

        valid_mask = ~np.isnan(mdata)
        valid = mdata[valid_mask]
        if len(valid) < 20:
            continue

        q = np.sum(valid == 0) / len(valid)
        pos = valid[valid > 0]
        if len(pos) < 10:
            continue

        params = fit_gamma_lmoments(pos)
        if params is None:
            continue
        alpha, beta = params['alpha'], params['beta']

        probs = np.full(len(mdata), np.nan)
        for i, val in enumerate(mdata):
            if np.isnan(val):
                continue
            if val == 0:
                probs[i] = q / 2 if q > 0 else 0.001
            else:
                probs[i] = q + (1 - q) * gamma_cdf(val, alpha, beta)

        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        ok = ~np.isnan(probs)
        if np.any(ok):
            ssi[indices[ok]] = norm_ppf_vec(probs[ok])

    return pd.Series(np.clip(ssi, -4.0, 4.0), index=monthly_discharge.index)


# ============================================================================
# MAIN
# ============================================================================

def main():
    start = datetime.now()

    SSI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load basin inventory
    # ------------------------------------------------------------------
    print("\n1. Loading basin inventory...")
    basin_inv = pd.read_csv(BASIN_INVENTORY)
    target_gages = set(basin_inv['GAGE_ID'].astype(str).str.zfill(8))
    gage_to_basin = dict(zip(
        basin_inv['GAGE_ID'].astype(str).str.zfill(8),
        basin_inv['basin_id']
    ))
    print(f"   Total basins: {len(basin_inv)}")

    # ------------------------------------------------------------------
    # 2. Read manifest (gage -> batch mapping)
    # ------------------------------------------------------------------
    print("\n2. Reading manifest...")
    manifest = pd.read_csv(MANIFEST_FILE)
    gage_to_batch = {}
    for _, row in manifest.iterrows():
        for g in str(row['sites']).split(','):
            g = g.strip()
            if g in target_gages:
                gage_to_batch[g] = row['raw_file']

    batch_files = sorted(set(gage_to_batch.values()))
    print(f"   Batches to process: {len(batch_files)}")

    # ------------------------------------------------------------------
    # 3. Load discharge + quality control -> Tier 1 basins
    # ------------------------------------------------------------------
    print("\n3. Loading discharge & quality-controlling gages...")
    all_discharge = {}       # gage_id -> DataFrame
    qc_metrics = {}          # gage_id -> {completeness, record_years}

    for bi, bf in enumerate(batch_files):
        if (bi + 1) % 20 == 0 or bi == 0:
            elapsed = (datetime.now() - start).total_seconds()
            rate = (bi + 1) / elapsed if elapsed > 0 else 0
            eta  = (len(batch_files) - bi - 1) / rate if rate > 0 else 0
            print(f"   Batch {bi+1}/{len(batch_files)} | ETA {eta/60:.1f} min | "
                  f"Gages loaded: {len(all_discharge)}")

        bpath = DISCHARGE_DIR / bf
        if not bpath.exists():
            continue

        site_data = parse_rdb_batch_file(str(bpath), target_gages)

        for gage_id, df in site_data.items():
            if len(df) < 30:
                continue

            first = df['datetime'].min()
            last  = df['datetime'].max()
            non_null = df['discharge'].notna().sum()
            span_days = (last - first).days + 1
            completeness = (non_null / span_days * 100) if span_days > 0 else 0
            record_yrs   = (last - first).days / 365.25

            qc_metrics[gage_id] = {
                'completeness': completeness,
                'record_years': record_yrs
            }

            # Keep discharge only for Tier 1 gages
            if completeness >= COMPLETENESS_TIER1 and record_yrs >= RECORD_LENGTH_TIER1:
                all_discharge[gage_id] = df

    tier1_gages = set(all_discharge.keys())
    print(f"\n   Total gages with data: {len(qc_metrics)}")
    print(f"   Tier 1 gages (>={COMPLETENESS_TIER1}% complete, >={RECORD_LENGTH_TIER1} yr): "
          f"{len(tier1_gages)}")

    # ------------------------------------------------------------------
    # 4. Monthly aggregation
    # ------------------------------------------------------------------
    print("\n4. Aggregating daily -> monthly discharge...")
    monthly = {}
    for gage_id, df in all_discharge.items():
        dfc = df.set_index('datetime')
        m = dfc['discharge'].resample('MS').mean()
        if len(m) >= 36:
            monthly[gage_id] = m

    print(f"   Gages with >=3 yr monthly: {len(monthly)}")

    # ------------------------------------------------------------------
    # 5. Compute SSI
    # ------------------------------------------------------------------
    print("\n5. Computing SSI (gamma + L-moments + per-month calibration)...")
    ssi_rows = []
    total = len(monthly)

    for idx, (gage_id, m_discharge) in enumerate(monthly.items()):
        if (idx + 1) % 500 == 0 or idx == 0:
            elapsed = (datetime.now() - start).total_seconds()
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta  = (total - idx - 1) / rate if rate > 0 else 0
            print(f"   [{idx+1}/{total}] ETA {eta:.0f}s")

        basin_id = gage_to_basin.get(gage_id)
        if basin_id is None:
            continue

        ssi_ts = {}
        for ts in TIMESCALES:
            ssi_ts[ts] = calculate_ssi(m_discharge, ts)

        # Build one row per date
        dates = m_discharge.index
        for d in dates:
            row = {'basin_id': basin_id, 'GAGE_ID': gage_id,
                   'date': d.strftime('%Y-%m-%d')}
            for ts in TIMESCALES:
                v = ssi_ts[ts].get(d, np.nan)
                row[f'SSI_{ts}'] = round(v, 4) if not np.isnan(v) else np.nan
            ssi_rows.append(row)

    ssi_df = pd.DataFrame(ssi_rows)
    print(f"   SSI records: {len(ssi_df):,}")
    print(f"   Basins: {ssi_df['basin_id'].nunique()}")

    # ------------------------------------------------------------------
    # 6. Aridity filter
    # ------------------------------------------------------------------
    print("\n6. Filtering persistently arid basins...")
    arid_basins = set()
    for basin_id, grp in ssi_df.groupby('basin_id'):
        freqs = []
        for ts in TIMESCALES:
            col = f'SSI_{ts}'
            vals = grp[col].dropna()
            if len(vals) > 0:
                freqs.append((vals <= DROUGHT_THRESHOLD).mean() * 100)
        if len(freqs) > 0 and np.mean(freqs) > ARIDITY_THRESHOLD:
            arid_basins.add(basin_id)

    print(f"   Persistently arid basins removed: {len(arid_basins)}")

    ssi_clean = ssi_df[~ssi_df['basin_id'].isin(arid_basins)].copy()
    print(f"   Non-arid Tier 1 basins: {ssi_clean['basin_id'].nunique()}")

    # ------------------------------------------------------------------
    # 7. Save outputs
    # ------------------------------------------------------------------
    print("\n7. Saving outputs...")
    ssi_clean.to_csv(SSI_OUTPUT_FILE, index=False)
    print(f"   SSI file: {SSI_OUTPUT_FILE}")
    print(f"   Records:  {len(ssi_clean):,}")

    tier1_list = (
        basin_inv[basin_inv['GAGE_ID'].astype(str).str.zfill(8).isin(tier1_gages)]
        .copy()
    )
    tier1_list = tier1_list[~tier1_list['basin_id'].isin(arid_basins)]
    tier1_list.to_csv(TIER1_OUTPUT, index=False)
    print(f"   Tier 1 non-arid list: {TIER1_OUTPUT}")
    print(f"   Basins: {len(tier1_list)}")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*80}")
    print("SSI COMPUTATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
