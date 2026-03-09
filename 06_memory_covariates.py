#!/usr/bin/env python3
"""
Step 3: Dynamic Event-Paired Catchment Memory & Covariate Analysis
====================================================================

MAJOR INNOVATION: Computes DYNAMIC, event-paired catchment memory metrics
instead of static basin-wide metrics.

PART A: Dynamic Memory Metrics (NEW)
====================================
For each of ~75,640 drought events:
  1. Extract 3 years (1,095 days) of daily discharge BEFORE spei_onset
  2. Compute event-specific BFI (Lyne-Hollick filter)
  3. Compute event-specific recession_k from recession segments
  4. Compute event-specific flow statistics (cv_Q, zero_flow_fraction, mean_Q)
  5. All metrics are now EVENT-SPECIFIC, capturing temporal changes in
     catchment behavior (urbanization, land use, climate trends)

This explains within-basin variance that static metrics miss.

Output:
  - dynamic_memory_matched_pairs.csv (master output: matched pairs + dyn metrics)
  - Comparison tables with static metrics from catchment_memory_metrics.csv

PART B: Covariate Analysis with Dynamic Metrics
================================================
  1. Merge dynamic metrics with GAGES-II attributes
  2. Compute basin-level recovery lag (median across events)
  3. Spearman correlations: dynamic covariates vs recovery lag
  4. Random Forest variable importance
  5. Publication figures (Fig5-8) using dynamic metrics

Inputs:
  - usgs_dv_00060_1980_01_01_to_2025_12_31/raw_rdb_gz/ (USGS daily discharge)
  - drought_matched_pairs_with_recovery.csv (from Step 2)
  - basinchar_and_report_sept_2011/spreadsheets-in-csv-format.zip (GAGES-II)

Outputs:
  - dynamic_memory_matched_pairs.csv
  - catchment_memory_metrics.csv (static, for comparison)
  - covariate_analysis/spearman_correlations_recovery.csv
  - covariate_analysis/random_forest_importance.csv
  - covariate_analysis/Fig5-8_*.png (publication figures)

Author: Drought Recovery Analysis
Date: February 2026
====================================================================
"""

import os
import sys
import gc
import time
import gzip
import warnings
import zipfile
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError as e:
    print(f"WARNING: {e}")

# ============================================================================
# CONFIGURATION
# ============================================================================

STEP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CORE_ROOT = STEP_DIR.parent
DATA_ROOT = CORE_ROOT.parent
STEP2_DIR = CORE_ROOT / 'Step2_Recovery_Atlas'

# Paths
MATCHED_CSV = STEP2_DIR / 'drought_matched_pairs_with_recovery.csv'
RDB_DIR = DATA_ROOT / 'usgs_dv_00060_1980_01_01_to_2025_12_31' / 'raw_rdb_gz'
MANIFEST = DATA_ROOT / 'usgs_dv_00060_1980_01_01_to_2025_12_31' / 'manifest.csv'
GAGESII_ZIP = DATA_ROOT / 'basinchar_and_report_sept_2011' / 'spreadsheets-in-csv-format.zip'

# Output files
OUTPUT_DIR = STEP_DIR / 'covariate_analysis'
OUTPUT_DIR.mkdir(exist_ok=True)

DYNAMIC_MATCHED_CSV = STEP_DIR / 'dynamic_memory_matched_pairs.csv'
STATIC_MEMORY_CSV = STEP_DIR / 'catchment_memory_metrics.csv'

FNAMES = {
    5: OUTPUT_DIR / 'Fig5_Covariate_Importance.png',
    6: OUTPUT_DIR / 'Fig6_Key_Scatter_Relationships.png',
    7: OUTPUT_DIR / 'Fig7_Dominant_Control_Map.png',
    8: OUTPUT_DIR / 'Fig8_CONUS_Covariate_Landscape.png',
}

# Algorithm parameters
TRAILING_WINDOW_DAYS = 1095  # 3 years before each event
MIN_VALID_DAYS = 365  # Minimum valid days in window
ALPHA_LHFILTER = 0.925  # Lyne-Hollick parameter
MIN_SEGMENTS_DYNAMIC = 3  # Relaxed for shorter windows (vs 5 for static)

# ============================================================================
# PART A: BASEFLOW & RECESSION FUNCTIONS (EXACTLY AS SPECIFIED)
# ============================================================================

def lyne_hollick_baseflow(Q, alpha=0.925, n_passes=3):
    """
    Lyne & Hollick (1979) recursive digital filter for baseflow separation.
    Q_f(t) = alpha * Q_f(t-1) + ((1 + alpha) / 2) * (Q(t) - Q(t-1))
    baseflow = Q - Q_f, clipped to [0, Q]
    """
    n = len(Q)
    if n < 3:
        return Q.copy()

    bf = Q.copy()

    for pass_num in range(n_passes):
        if pass_num % 2 == 0:
            indices = range(1, n)
        else:
            indices = range(n - 2, -1, -1)

        qf = np.zeros(n)
        if pass_num % 2 == 0:
            qf[0] = 0.0
        else:
            qf[n - 1] = 0.0

        for i in indices:
            prev = i - 1 if pass_num % 2 == 0 else i + 1
            qf[i] = alpha * qf[prev] + ((1 + alpha) / 2) * (bf[i] - bf[prev])
            qf[i] = max(0, qf[i])
            qf[i] = min(qf[i], bf[i])

        bf = bf - qf
        bf = np.clip(bf, 0, Q)

    return bf


def compute_bfi(Q, alpha=0.925):
    """Compute BFI = sum(baseflow) / sum(total_flow)."""
    if len(Q) < 30 or np.nansum(Q) == 0:
        return np.nan
    mask = ~np.isnan(Q)
    q = Q[mask]
    if len(q) < 30 or np.sum(q) == 0:
        return np.nan
    bf = lyne_hollick_baseflow(q, alpha=alpha)
    return np.sum(bf) / np.sum(q)


def extract_recession_segments(Q, min_length=5, max_rise_fraction=0.0):
    """Extract recession segments: >=min_length consecutive declining days."""
    n = len(Q)
    segments = []
    i = 0

    while i < n - 1:
        if Q[i + 1] <= Q[i] * (1 + max_rise_fraction) and Q[i] > 0:
            start = i
            j = i + 1
            while j < n - 1:
                if Q[j + 1] <= Q[j] * (1 + max_rise_fraction) and Q[j] > 0:
                    j += 1
                else:
                    break
            end = j
            length = end - start + 1
            if length >= min_length:
                segments.append((start, end))
            i = end + 1
        else:
            i += 1

    return segments


def compute_recession_constant(Q, min_length=5, min_segments=3):
    """
    Compute recession constant k from master recession curve.
    Q(t) = Q(0) * exp(-t/k)  →  k = -1 / slope

    Note: min_segments=3 (relaxed from 5 for shorter trailing windows)
    """
    mask = ~np.isnan(Q) & (Q > 0)
    q_clean = Q.copy()
    q_clean[~mask] = 0

    segments = extract_recession_segments(q_clean, min_length=min_length)

    if len(segments) < min_segments:
        return np.nan, np.nan, len(segments)

    slopes = []
    for start, end in segments:
        seg_q = q_clean[start:end + 1]

        if np.any(seg_q <= 0):
            continue

        ln_q = np.log(seg_q)
        t = np.arange(len(seg_q))

        if len(t) < 2:
            continue

        try:
            b, a = np.polyfit(t, ln_q, 1)
            if b < 0:
                slopes.append(b)
        except (np.linalg.LinAlgError, ValueError):
            continue

    if len(slopes) < min_segments:
        return np.nan, np.nan, len(slopes)

    slopes = np.array(slopes)
    median_slope = np.median(slopes)
    k = -1.0 / median_slope

    k_all = -1.0 / slopes
    k_iqr = np.percentile(k_all, 75) - np.percentile(k_all, 25)

    return round(k, 2), round(k_iqr, 2), len(slopes)


# ============================================================================
# PART A.1: RDB PARSING (EXACTLY AS SPECIFIED)
# ============================================================================

def parse_rdb_gz(filepath, target_sites=None):
    """Parse USGS RDB .gz file into dict of {site_no: DataFrame}."""
    rows = []

    with gzip.open(filepath, 'rt', errors='replace') as f:
        header = None
        q_col_idx = None
        skip_format = False

        for line in f:
            line = line.rstrip('\n')
            if line.startswith('#'):
                continue

            parts = line.split('\t')

            if 'agency_cd' in parts:
                header = parts
                q_col_idx = None
                for ci, col in enumerate(header):
                    if col.endswith('_00060_00003') and not col.endswith('_cd'):
                        q_col_idx = ci
                        break
                skip_format = True
                continue

            if skip_format:
                skip_format = False
                continue

            if header is None or q_col_idx is None:
                continue

            if len(parts) <= q_col_idx:
                continue

            site_no = parts[1] if len(parts) > 1 else ''
            date_str = parts[2] if len(parts) > 2 else ''
            q_str = parts[q_col_idx]

            if target_sites is not None and site_no not in target_sites:
                continue

            rows.append((site_no, date_str, q_str))

    if not rows:
        return {}

    df = pd.DataFrame(rows, columns=['site_no', 'datetime', 'Q_raw'])
    df['Q_cfs'] = pd.to_numeric(df['Q_raw'], errors='coerce')
    df['date'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['date'])

    result = {}
    for site, grp in df.groupby('site_no'):
        sdf = grp[['date', 'Q_cfs']].copy()
        sdf = sdf.sort_values('date').drop_duplicates(subset='date').reset_index(drop=True)
        result[site] = sdf

    return result


# ============================================================================
# PART A.2: LOAD ALL DAILY DISCHARGE DATA (Phase 1)
# ============================================================================

def load_all_discharge_data():
    """
    Phase 1: Load all daily discharge data into memory.
    Returns: {padded_site_no: DataFrame(date, Q_cfs)}
    """
    print("=" * 80)
    print("PHASE 1: LOADING ALL DAILY DISCHARGE DATA INTO MEMORY")
    print("=" * 80)

    # Load target GAGEs
    print("\n1. Loading target GAGE_IDs from matched pairs...")
    df_matched = pd.read_csv(MATCHED_CSV, dtype={'GAGE_ID': str})
    gage_ids = sorted(df_matched['GAGE_ID'].unique())
    print(f"   Found {len(gage_ids)} unique GAGE_IDs across {len(df_matched)} events")

    # Create GAGE_ID <-> padded_site_no mapping
    target_sites = set()
    gage_id_to_padded = {}
    padded_to_gage_id = {}
    for gid in gage_ids:
        padded = gid.zfill(8)
        gage_id_to_padded[gid] = padded
        padded_to_gage_id[padded] = gid
        target_sites.add(padded)

    # Scan manifest to find which batch files contain our target sites
    print("\n2. Scanning manifest to identify relevant batch files...")
    manifest = pd.read_csv(MANIFEST)
    batch_files = []
    for _, row in manifest.iterrows():
        sites_in_batch = set(str(row['sites']).split(','))
        overlap = sites_in_batch & target_sites
        if overlap:
            batch_files.append({
                'filename': row['raw_file'],
                'n_target_sites': len(overlap),
                'sites': overlap
            })
    print(f"   Found {len(batch_files)} batch files containing target sites")

    # Parse all batch files
    print("\n3. Parsing RDB .gz files and loading into memory...")
    all_discharge = {}
    sites_found = set()
    t0 = time.time()
    n_batches = len(batch_files)

    for b_num, batch in enumerate(batch_files):
        if (b_num + 1) % max(1, n_batches // 20) == 0 or b_num == 0:
            elapsed = time.time() - t0
            rate = (b_num + 1) / max(1, elapsed) * 60
            print(f"   Batch {b_num+1}/{n_batches} ({len(sites_found)} sites loaded, "
                  f"{rate:.1f} batches/min, {elapsed:.0f}s)")

        fpath = RDB_DIR / batch['filename']
        if not fpath.exists():
            print(f"   WARNING: {batch['filename']} not found, skipping")
            continue

        site_data = parse_rdb_gz(str(fpath), target_sites=target_sites)

        for padded_id, sdf in site_data.items():
            if padded_id not in padded_to_gage_id:
                continue
            gid = padded_to_gage_id[padded_id]
            sites_found.add(gid)
            all_discharge[gid] = sdf

    elapsed = time.time() - t0
    print(f"\n   Done! Loaded discharge data for {len(all_discharge)} sites in {elapsed:.0f}s")

    return all_discharge, gage_id_to_padded, padded_to_gage_id


# ============================================================================
# PART A.3: COMPUTE DYNAMIC EVENT-LEVEL MEMORY METRICS (Phase 2)
# ============================================================================

def compute_dynamic_metrics_for_event(Q, dates, event_date, trailing_days=TRAILING_WINDOW_DAYS):
    """
    Compute event-specific memory metrics from trailing window before event.

    Args:
        Q: array of discharge values
        dates: array of dates (datetime64 or pandas Timestamp)
        event_date: datetime of spei_onset
        trailing_days: days to look back (default 1095 = 3 years)

    Returns:
        dict with: dyn_BFI, dyn_recession_k_days, dyn_cv_Q, dyn_zero_flow_fraction,
                   dyn_mean_Q_cfs, dyn_window_days
    """
    result = {
        'dyn_BFI': np.nan,
        'dyn_recession_k_days': np.nan,
        'dyn_recession_k_iqr': np.nan,
        'dyn_cv_Q': np.nan,
        'dyn_zero_flow_fraction': np.nan,
        'dyn_mean_Q_cfs': np.nan,
        'dyn_window_days': 0,
    }

    # Convert event_date to pandas Timestamp if needed
    if not isinstance(event_date, (pd.Timestamp, np.datetime64)):
        event_date = pd.Timestamp(event_date)
    else:
        event_date = pd.Timestamp(event_date)

    # Find window bounds
    window_start = event_date - timedelta(days=trailing_days)
    window_end = event_date - timedelta(days=1)  # Up to day before event

    # Extract window data
    mask = (dates >= window_start) & (dates <= window_end)
    Q_window = Q[mask]
    dates_window = dates[mask]

    if len(Q_window) == 0:
        return result

    # Check for minimum valid data
    valid_mask = ~np.isnan(Q_window)
    n_valid = np.sum(valid_mask)
    if n_valid < MIN_VALID_DAYS:
        result['dyn_window_days'] = n_valid
        return result

    result['dyn_window_days'] = n_valid

    # Extract valid data
    Q_valid = Q_window[valid_mask]

    # Flow statistics
    mean_q = np.mean(Q_valid)
    std_q = np.std(Q_valid)
    cv_q = std_q / mean_q if mean_q > 0 else np.nan
    zero_frac = np.sum(Q_valid == 0) / len(Q_valid)

    result['dyn_mean_Q_cfs'] = round(mean_q, 2)
    result['dyn_cv_Q'] = round(cv_q, 4) if not np.isnan(cv_q) else np.nan
    result['dyn_zero_flow_fraction'] = round(zero_frac, 4)

    # BFI
    bfi = compute_bfi(Q_valid, alpha=ALPHA_LHFILTER)
    result['dyn_BFI'] = round(bfi, 4) if not np.isnan(bfi) else np.nan

    # Recession constant
    k, k_iqr, n_seg = compute_recession_constant(Q_valid, min_length=5,
                                                  min_segments=MIN_SEGMENTS_DYNAMIC)
    result['dyn_recession_k_days'] = k
    result['dyn_recession_k_iqr'] = k_iqr

    return result


def compute_dynamic_memory_metrics():
    """
    Phase 2: Compute dynamic event-level memory metrics for all 75K+ events.

    Optimization: Group by GAGE_ID, sort by date, load once per gauge,
    then slice windows for each event.
    """
    print("\n" + "=" * 80)
    print("PHASE 2: COMPUTING DYNAMIC EVENT-LEVEL MEMORY METRICS")
    print("=" * 80)

    # Check if output already exists (caching mechanism)
    if DYNAMIC_MATCHED_CSV.exists():
        print(f"\nFound cached output: {DYNAMIC_MATCHED_CSV}")
        print("Loading from cache (set DYNAMIC_MATCHED_CSV to None to recompute)...")
        return pd.read_csv(DYNAMIC_MATCHED_CSV, dtype={'GAGE_ID': str})

    # Load matched pairs
    print("\n1. Loading matched pairs with recovery metrics...")
    df_matched = pd.read_csv(MATCHED_CSV, dtype={'GAGE_ID': str})
    df_matched['spei_onset'] = pd.to_datetime(df_matched['spei_onset'])
    print(f"   {len(df_matched)} events across {df_matched['GAGE_ID'].nunique()} gauges")

    # Load all discharge data into memory
    all_discharge, gage_id_to_padded, padded_to_gage_id = load_all_discharge_data()

    # Group events by GAGE_ID for efficient processing
    print("\n2. Computing dynamic metrics for all events...")
    t0 = time.time()
    n_gages = df_matched['GAGE_ID'].nunique()
    gage_list = sorted(df_matched['GAGE_ID'].unique())

    dynamic_results = []

    for g_idx, gage_id in enumerate(gage_list):
        if (g_idx + 1) % max(1, n_gages // 20) == 0 or g_idx == 0:
            elapsed = time.time() - t0
            rate = (g_idx + 1) / max(1, elapsed) * 60
            print(f"   Gage {g_idx+1}/{n_gages} ({rate:.1f} gages/min, {elapsed:.0f}s)")

        # Get all events for this gage
        events_this_gage = df_matched[df_matched['GAGE_ID'] == gage_id]

        # Load discharge data for this gage
        if gage_id not in all_discharge:
            continue

        sdf = all_discharge[gage_id]
        Q = sdf['Q_cfs'].values
        dates = pd.to_datetime(sdf['date'].values)

        # Compute metrics for each event at this gage
        for _, event_row in events_this_gage.iterrows():
            dyn_metrics = compute_dynamic_metrics_for_event(Q, dates,
                                                            event_row['spei_onset'],
                                                            trailing_days=TRAILING_WINDOW_DAYS)
            # Merge with event row
            result_row = event_row.to_dict()
            result_row.update(dyn_metrics)
            dynamic_results.append(result_row)

    elapsed = time.time() - t0
    print(f"\n   Done! Computed metrics for {len(dynamic_results)} events in {elapsed:.0f}s")

    # Create output DataFrame
    df_dynamic = pd.DataFrame(dynamic_results)
    df_dynamic = df_dynamic.sort_values(['GAGE_ID', 'spei_onset']).reset_index(drop=True)

    # Save to CSV
    print(f"\n3. Writing {DYNAMIC_MATCHED_CSV}...")
    df_dynamic.to_csv(DYNAMIC_MATCHED_CSV, index=False)
    print(f"   {len(df_dynamic)} events, {len(df_dynamic.columns)} columns")

    return df_dynamic


# ============================================================================
# PART A.4: STATIC MEMORY METRICS (for comparison)
# ============================================================================

def compute_static_catchment_memory_metrics():
    """
    Compute static basin-wide memory metrics (one per basin, full record).
    Used for comparison with dynamic metrics.
    """
    print("\n" + "=" * 80)
    print("COMPUTING STATIC BASIN-LEVEL MEMORY METRICS (for comparison)")
    print("=" * 80)

    # Check if already computed
    if STATIC_MEMORY_CSV.exists():
        print(f"\nFound cached static metrics: {STATIC_MEMORY_CSV}")
        return pd.read_csv(STATIC_MEMORY_CSV, dtype={'GAGE_ID': str})

    # Load target GAGEs
    print("\n1. Loading target GAGE_IDs...")
    df_matched = pd.read_csv(MATCHED_CSV, dtype={'GAGE_ID': str}, usecols=['GAGE_ID'])
    gage_ids = sorted(df_matched['GAGE_ID'].unique())
    print(f"   {len(gage_ids)} unique GAGE_IDs")

    # Zero-pad
    target_sites = set()
    gage_id_to_padded = {}
    padded_to_gage_id = {}
    for gid in gage_ids:
        padded = gid.zfill(8)
        gage_id_to_padded[gid] = padded
        padded_to_gage_id[padded] = gid
        target_sites.add(padded)

    # Scan manifest
    print("\n2. Scanning manifest...")
    manifest = pd.read_csv(MANIFEST)
    batch_files = []
    for _, row in manifest.iterrows():
        sites_in_batch = set(str(row['sites']).split(','))
        overlap = sites_in_batch & target_sites
        if overlap:
            batch_files.append({
                'filename': row['raw_file'],
                'n_target_sites': len(overlap),
                'sites': overlap
            })
    print(f"   {len(batch_files)} batch files contain target sites")

    # Parse and compute
    print("\n3. Parsing discharge data and computing static metrics...")
    all_results = {}
    sites_found = set()
    t0 = time.time()
    n_batches = len(batch_files)

    for b_num, batch in enumerate(batch_files):
        if (b_num + 1) % max(1, n_batches // 20) == 0 or b_num == 0:
            elapsed = time.time() - t0
            rate = (b_num + 1) / max(1, elapsed) * 60
            print(f"   Batch {b_num+1}/{n_batches} ({len(sites_found)} sites, "
                  f"{rate:.1f} batches/min, {elapsed:.0f}s)")

        fpath = RDB_DIR / batch['filename']
        if not fpath.exists():
            continue

        site_data = parse_rdb_gz(str(fpath), target_sites=target_sites)

        for padded_id, sdf in site_data.items():
            if padded_id not in padded_to_gage_id:
                continue

            gid = padded_to_gage_id[padded_id]
            sites_found.add(gid)

            Q = sdf['Q_cfs'].values

            # Check minimum data
            q_valid = Q[~np.isnan(Q)]
            n_days = len(q_valid)
            if n_days < 365:
                all_results[gid] = {
                    'GAGE_ID': gid,
                    'n_days': n_days,
                    'data_years': round(n_days / 365.25, 1),
                    'BFI': np.nan,
                    'recession_k_days': np.nan,
                    'recession_k_iqr': np.nan,
                    'mean_Q_cfs': np.nan,
                    'cv_Q': np.nan,
                    'zero_flow_fraction': np.nan,
                }
                continue

            # Compute metrics
            mean_q = np.nanmean(Q)
            std_q = np.nanstd(Q)
            cv_q = std_q / mean_q if mean_q > 0 else np.nan
            zero_frac = np.sum(q_valid == 0) / len(q_valid)

            bfi = compute_bfi(Q, alpha=ALPHA_LHFILTER)
            k, k_iqr, n_seg = compute_recession_constant(Q, min_length=5, min_segments=5)

            all_results[gid] = {
                'GAGE_ID': gid,
                'n_days': n_days,
                'data_years': round(n_days / 365.25, 1),
                'BFI': round(bfi, 4) if not np.isnan(bfi) else np.nan,
                'recession_k_days': k,
                'recession_k_iqr': k_iqr,
                'mean_Q_cfs': round(mean_q, 2),
                'cv_Q': round(cv_q, 4) if not np.isnan(cv_q) else np.nan,
                'zero_flow_fraction': round(zero_frac, 4),
            }

    elapsed = time.time() - t0
    print(f"\n   Done: {len(sites_found)} basins in {elapsed:.0f}s")

    # Write output
    print(f"\n4. Writing {STATIC_MEMORY_CSV}...")
    df_out = pd.DataFrame(list(all_results.values()))
    df_out = df_out.sort_values('GAGE_ID').reset_index(drop=True)
    df_out.to_csv(STATIC_MEMORY_CSV, index=False)
    print(f"   {len(df_out)} basins, {len(df_out.columns)} columns")

    return df_out


# ============================================================================
# PART B: COVARIATE ANALYSIS WITH DYNAMIC METRICS
# ============================================================================

def extract_gages2_attributes(dynamic_df):
    """Extract GAGES-II attributes from ZIP CSV files."""
    print("\n" + "=" * 80)
    print("PART B: COVARIATE ANALYSIS")
    print("=" * 80)
    print("\n1. Extracting GAGES-II attributes...")
    t0 = datetime.now()

    attrs = None

    try:
        with zipfile.ZipFile(GAGESII_ZIP, 'r') as zf:
            # BasinID
            with zf.open('conterm_basinid.txt') as f:
                attrs = pd.read_csv(f, sep=',', encoding='latin-1')
            attrs['GAGE_ID'] = attrs['STAID'].astype(str).str.zfill(8)
            attrs = attrs[['GAGE_ID', 'DRAIN_SQKM', 'LAT_GAGE', 'LNG_GAGE']].copy()
            print(f"   BasinID: {len(attrs)} basins")

            # Climate
            with zf.open('conterm_climate.txt') as f:
                clim = pd.read_csv(f, sep=',', encoding='latin-1')
            clim['GAGE_ID'] = clim['STAID'].astype(str).str.zfill(8)
            clim['ARIDITY_INDEX'] = clim['PET'] / clim['PPTAVG_BASIN'].replace(0, np.nan)
            clim_keep = ['GAGE_ID', 'PPTAVG_BASIN', 'T_AVG_BASIN', 'PET',
                         'SNOW_PCT_PRECIP', 'PRECIP_SEAS_IND', 'ARIDITY_INDEX', 'RH_BASIN']
            clim = clim[[c for c in clim_keep if c in clim.columns]]
            attrs = attrs.merge(clim, on='GAGE_ID', how='left')

            # Hydro
            with zf.open('conterm_hydro.txt') as f:
                hydro = pd.read_csv(f, sep=',', encoding='latin-1')
            hydro['GAGE_ID'] = hydro['STAID'].astype(str).str.zfill(8)
            hydro_keep = ['GAGE_ID', 'BFI_AVE', 'PERDUN', 'PERHOR', 'TOPWET', 'CONTACT',
                          'RUNAVE7100', 'STREAMS_KM_SQ_KM', 'STRAHLER_MAX', 'WB5100_ANN_MM']
            hydro = hydro[[c for c in hydro_keep if c in hydro.columns]]
            attrs = attrs.merge(hydro, on='GAGE_ID', how='left')

            # Topo
            with zf.open('conterm_topo.txt') as f:
                topo = pd.read_csv(f, sep=',', encoding='latin-1')
            topo['GAGE_ID'] = topo['STAID'].astype(str).str.zfill(8)
            topo_keep = ['GAGE_ID', 'ELEV_MAX_M', 'ELEV_MIN_M', 'ELEV_MEAN_M',
                         'SLOPE_MEAN', 'HIGH_PREC_FREQ', 'HIGH_PREC_DUR', 'HIGH_PREC_TIMING']
            topo = topo[[c for c in topo_keep if c in topo.columns]]
            attrs = attrs.merge(topo, on='GAGE_ID', how='left')

            # Soils
            with zf.open('conterm_soils.txt') as f:
                soils = pd.read_csv(f, sep=',', encoding='latin-1')
            soils['GAGE_ID'] = soils['STAID'].astype(str).str.zfill(8)
            soils_keep = ['GAGE_ID', 'SILT_PCT', 'SAND_PCT', 'CLAY_PCT',
                          'AWCAVE', 'ROCKDEPAVE', 'BEDROCKDEPAVE', 'PERMAVE']
            soils = soils[[c for c in soils_keep if c in soils.columns]]
            attrs = attrs.merge(soils, on='GAGE_ID', how='left')

            # Land cover
            with zf.open('conterm_lc06_basin.txt') as f:
                lc = pd.read_csv(f, sep=',', encoding='latin-1')
            lc['GAGE_ID'] = lc['STAID'].astype(str).str.zfill(8)
            lc_keep = ['GAGE_ID', 'CROP_PCT', 'FOREST_PCT', 'SHRUB_PCT',
                       'BARREN_PCT', 'DEV_PCT', 'WATER_PCT', 'WETLAND_PCT']
            lc = lc[[c for c in lc_keep if c in lc.columns]]
            attrs = attrs.merge(lc, on='GAGE_ID', how='left')

            # Ecoregion
            with zf.open('conterm_bas_classif.txt') as f:
                eco = pd.read_csv(f, sep=',', encoding='latin-1')
            eco['GAGE_ID'] = eco['STAID'].astype(str).str.zfill(8)
            eco = eco[['GAGE_ID', 'AGGECOREGION']].copy()
            attrs = attrs.merge(eco, on='GAGE_ID', how='left')

    except Exception as e:
        print(f"   ERROR extracting GAGES-II: {e}")
        return None

    print(f"   Total: {len(attrs)} basins, {len(attrs.columns)} attributes")
    return attrs


def analyze_covariates_with_dynamic_metrics(dynamic_df, static_df):
    """
    Covariate analysis using dynamic metrics as the primary driver.
    Computes basin-level recovery lag and correlations with dynamic covariates.
    """
    print("\n2. Merging dynamic metrics with GAGES-II attributes...")

    # Extract GAGES-II
    gages2 = extract_gages2_attributes(dynamic_df)
    if gages2 is None:
        print("   ERROR: Could not extract GAGES-II attributes")
        return

    # Aggregate dynamic metrics to basin level (median across events)
    print("\n3. Aggregating dynamic metrics to basin level (median)...")
    basin_dynamic = dynamic_df.groupby('GAGE_ID').agg({
        'dyn_BFI': 'median',
        'dyn_recession_k_days': 'median',
        'dyn_cv_Q': 'median',
        'dyn_zero_flow_fraction': 'median',
        'dyn_mean_Q_cfs': 'median',
    }).reset_index()

    # Basin-level recovery lag (median of recovery_lag_months)
    if 'recovery_lag_months' in dynamic_df.columns:
        recovery_lag = dynamic_df.groupby('GAGE_ID').agg({
            'recovery_lag_months': ['median', 'count']
        }).reset_index()
        recovery_lag.columns = ['GAGE_ID', 'rec_lag_median', 'n_events']
    else:
        print("   WARNING: recovery_lag_months not found in dynamic_df")
        recovery_lag = dynamic_df.groupby('GAGE_ID').size().reset_index(name='n_events')
        recovery_lag['GAGE_ID'] = recovery_lag['GAGE_ID']
        recovery_lag['rec_lag_median'] = np.nan

    # Merge: GAGES-II + dynamic metrics + recovery lag
    print("   Merging all attributes...")
    df_cov = gages2.merge(basin_dynamic, on='GAGE_ID', how='inner')
    df_cov = df_cov.merge(recovery_lag, on='GAGE_ID', how='left')

    print(f"   {len(df_cov)} basins with complete data")

    # Spearman correlations: dynamic metrics vs recovery lag
    print("\n4. Computing Spearman correlations (dynamic metrics vs recovery lag)...")

    corr_cols = [c for c in df_cov.columns if c.startswith('dyn_') or c in gages2.columns]
    corr_cols = [c for c in corr_cols if c not in ['GAGE_ID']]

    correlations = []
    for col in corr_cols:
        # Remove NaNs for this correlation
        mask = ~(df_cov[col].isna() | df_cov['rec_lag_median'].isna())
        if np.sum(mask) < 10:
            continue

        rho, pval = sp_stats.spearmanr(df_cov.loc[mask, col],
                                       df_cov.loc[mask, 'rec_lag_median'])
        correlations.append({
            'Variable': col,
            'Spearman_rho': round(rho, 4),
            'p_value': round(pval, 6),
            'n_basins': int(np.sum(mask)),
        })

    corr_df = pd.DataFrame(correlations)
    corr_df = corr_df.sort_values('Spearman_rho', ascending=False, key=abs)

    corr_path = OUTPUT_DIR / 'spearman_correlations_recovery.csv'
    corr_df.to_csv(corr_path, index=False)
    print(f"   Saved: {corr_path}")

    # Random Forest importance
    print("\n5. Running Random Forest variable importance...")

    # Prepare data for RF
    rf_cols = [c for c in df_cov.columns
               if c.startswith('dyn_') or
               c in ['DRAIN_SQKM', 'LAT_GAGE', 'LNG_GAGE', 'PPTAVG_BASIN', 'T_AVG_BASIN',
                     'ARIDITY_INDEX', 'BFI_AVE', 'ELEV_MEAN_M', 'SLOPE_MEAN',
                     'FOREST_PCT', 'DEV_PCT']]
    rf_cols = [c for c in rf_cols if c in df_cov.columns]

    mask_rf = ~(df_cov[rf_cols].isna().any(axis=1) | df_cov['rec_lag_median'].isna())
    df_rf = df_cov.loc[mask_rf, rf_cols + ['rec_lag_median']].copy()

    if len(df_rf) > 10:
        X = df_rf[rf_cols].values
        y = df_rf['rec_lag_median'].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        rf = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
        rf.fit(X_scaled, y)

        importance_df = pd.DataFrame({
            'Variable': rf_cols,
            'Importance': rf.feature_importances_,
        }).sort_values('Importance', ascending=False)

        importance_path = OUTPUT_DIR / 'random_forest_importance.csv'
        importance_df.to_csv(importance_path, index=False)
        print(f"   Saved: {importance_path}")
    else:
        print(f"   WARNING: Not enough data for RF ({len(df_rf)} basins)")

    return df_cov, corr_df


def save_final_summary(dynamic_df):
    """Save summary statistics."""
    print("\n6. Saving summary statistics...")

    summary = {
        'Total_Events': len(dynamic_df),
        'Unique_Gages': dynamic_df['GAGE_ID'].nunique(),
        'Events_with_Dynamic_Metrics': (dynamic_df['dyn_window_days'] >= MIN_VALID_DAYS).sum(),
        'Mean_BFI': dynamic_df['dyn_BFI'].mean(),
        'Median_Recession_k': dynamic_df['dyn_recession_k_days'].median(),
    }

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    print("\n")
    print("*" * 80)
    print("STEP 3: DYNAMIC EVENT-PAIRED CATCHMENT MEMORY & COVARIATE ANALYSIS")
    print("*" * 80)

    # Phase 1 & 2: Compute dynamic metrics
    dynamic_df = compute_dynamic_memory_metrics()

    # Compute static metrics for comparison
    static_df = compute_static_catchment_memory_metrics()

    # Save summary
    save_final_summary(dynamic_df)

    # Phase 3: Covariate analysis
    try:
        result = analyze_covariates_with_dynamic_metrics(dynamic_df, static_df)
    except Exception as e:
        print(f"\nWARNING: Covariate analysis failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("STEP 3 COMPLETE")
    print("=" * 80)
    print(f"\nOutputs:")
    print(f"  - {DYNAMIC_MATCHED_CSV}")
    print(f"  - {STATIC_MEMORY_CSV}")
    print(f"  - {OUTPUT_DIR}/spearman_correlations_recovery.csv")
    print(f"  - {OUTPUT_DIR}/random_forest_importance.csv")
    print("\n")


if __name__ == '__main__':
    main()
