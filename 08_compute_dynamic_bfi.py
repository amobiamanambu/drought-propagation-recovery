#!/usr/bin/env python3
"""
Compute dynamic BFI (3-year trailing window) for buffered and independently
initiated SSI drought events.

Reads:
  - Fix_core/all_events_with_dynamic_bfi.csv  (242,912 events; propagated already have BFI)
  - usgs_dv_00060_1980_01_01_to_2025_12_31/raw_rdb_gz/*.rdb.gz  (daily discharge)

Writes:
  - Fix_core/all_events_with_dynamic_bfi.csv  (updated with BFI for buffered + independent)

Run: python Fix_core/compute_dynamic_bfi_all_events.py
Expected runtime: ~10-15 min on a laptop with 8+ GB RAM
"""

import pandas as pd
import numpy as np
import gzip, time, gc
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).resolve().parent.parent
RDB_DIR = BASE / 'usgs_dv_00060_1980_01_01_to_2025_12_31' / 'raw_rdb_gz'
CSV_PATH = BASE / 'Fix_core' / 'all_events_with_dynamic_bfi.csv'


def lyne_hollick_bf(Q_arr, alpha=0.925, n_passes=3):
    """Lyne-Hollick digital filter for baseflow separation."""
    q = Q_arr.copy()
    for p in range(n_passes):
        qf = np.zeros_like(q)
        if p % 2 == 0:
            for i in range(1, len(q)):
                qf[i] = max(0.0, alpha * qf[i-1] + (1+alpha)/2 * (q[i] - q[i-1]))
        else:
            for i in range(len(q)-2, -1, -1):
                qf[i] = max(0.0, alpha * qf[i+1] + (1+alpha)/2 * (q[i] - q[i+1]))
        q = np.maximum(q - qf, 0.0)
    return q


def compute_metrics_from_window(Q_vals):
    """Compute BFI and recession metrics from a discharge array."""
    bf = lyne_hollick_bf(Q_vals)
    total = Q_vals.sum()
    bfi = bf.sum() / total if total > 0 else np.nan

    dQ = np.diff(Q_vals)
    rec_mask = dQ < 0
    if rec_mask.sum() > 10:
        Qr = Q_vals[:-1][rec_mask]
        dQr = dQ[rec_mask]
        with np.errstate(divide='ignore', invalid='ignore'):
            kv = -Qr / dQr
            kv = kv[(kv > 0) & (kv < 1000) & np.isfinite(kv)]
        rec_k = float(np.median(kv)) if len(kv) > 0 else np.nan
    else:
        rec_k = np.nan

    cv = float(np.std(Q_vals) / np.mean(Q_vals)) if np.mean(Q_vals) > 0 else np.nan
    zf = float((Q_vals == 0).sum() / len(Q_vals))

    return {
        'dyn_BFI': bfi,
        'dyn_recession_k_days': rec_k,
        'dyn_cv_Q': cv,
        'dyn_zero_flow_fraction': zf,
        'dyn_mean_Q_cfs': float(np.mean(Q_vals)),
        'dyn_window_days': len(Q_vals),
    }


def main():
    t0 = time.time()

    # ── Load events ──
    print("Loading events ...")
    df = pd.read_csv(CSV_PATH, dtype={'GAGE_ID': str}, low_memory=False)
    mask = df['dyn_BFI'].isna() & df['event_type'].isin(['buffered', 'independent_ssi'])
    needs = df.loc[mask, ['GAGE_ID', 'onset_date']].copy()
    needs['gage_stripped'] = needs['GAGE_ID'].str.lstrip('0')
    print(f"  {len(needs)} events need BFI across {needs['gage_stripped'].nunique()} sites")

    if len(needs) == 0:
        print("  All events already have BFI. Nothing to do.")
        return

    # Build lookup: site -> list of (df_index, onset_date)
    site_events = defaultdict(list)
    for idx, row in needs.iterrows():
        site_events[row['gage_stripped']].append((idx, row['onset_date']))
    sites_needed = set(site_events.keys())

    # ── Phase 1: Load all daily discharge ──
    print(f"\nPhase 1: Loading daily discharge for {len(sites_needed)} sites ...")
    t1 = time.time()

    all_Q = defaultdict(list)  # site -> [(date_str, Q), ...]
    rdb_files = sorted(RDB_DIR.glob('*.rdb.gz'))
    print(f"  {len(rdb_files)} RDB batch files")

    for fi, fp in enumerate(rdb_files):
        with gzip.open(fp, 'rt', errors='replace') as f:
            lines = f.readlines()

        header_idx = None
        for j, line in enumerate(lines):
            if line.startswith('agency_cd'):
                header_idx = j
                break
        if header_idx is None:
            continue

        headers = lines[header_idx].strip().split('\t')
        try:
            site_col = headers.index('site_no')
            date_col = headers.index('datetime')
        except ValueError:
            continue

        val_col = None
        for k, h in enumerate(headers):
            if '00060' in h and '00003' in h and '_cd' not in h:
                val_col = k
                break
        if val_col is None:
            for k, h in enumerate(headers):
                if '00060' in h and '_cd' not in h:
                    val_col = k
                    break
        if val_col is None:
            continue

        for line in lines[header_idx + 2:]:
            parts = line.split('\t')
            if len(parts) <= max(site_col, date_col, val_col):
                continue
            site = parts[site_col].lstrip('0')
            if site not in sites_needed:
                continue
            try:
                q = float(parts[val_col])
                if q >= 0:
                    all_Q[site].append((parts[date_col], q))
            except (ValueError, IndexError):
                continue

        del lines
        if (fi + 1) % 25 == 0:
            print(f"    File {fi+1}/{len(rdb_files)}, {len(all_Q)} sites, {time.time()-t1:.0f}s")

    print(f"  Loaded {len(all_Q)} sites in {time.time()-t1:.0f}s")

    # ── Phase 2: Compute metrics per site ──
    print(f"\nPhase 2: Computing dynamic metrics ...")
    t2 = time.time()
    computed = 0
    skipped = 0
    results = {}

    for si, (site, records) in enumerate(all_Q.items()):
        if len(records) < 365:
            continue

        # Build time series
        dates_raw = [r[0] for r in records]
        flows_raw = [r[1] for r in records]
        dates = pd.to_datetime(dates_raw, errors='coerce')
        flows = np.array(flows_raw)

        valid = ~pd.isnull(dates)
        dates = dates[valid]
        flows = flows[valid]

        if len(dates) < 365:
            continue

        order = np.argsort(dates)
        dates = dates[order]
        flows = flows[order]

        _, unique_idx = np.unique(dates, return_index=True)
        dates = dates[unique_idx]
        flows = flows[unique_idx]

        Q_series = pd.Series(flows, index=dates)

        for df_idx, onset in site_events.get(site, []):
            if pd.isna(onset):
                skipped += 1
                continue
            end = pd.Timestamp(onset)
            start = end - pd.Timedelta(days=1095)
            w = Q_series.loc[start:end]

            if len(w) < 365:
                skipped += 1
                continue

            Q_vals = w.values.astype(float)
            metrics = compute_metrics_from_window(Q_vals)
            results[df_idx] = metrics
            computed += 1

        if (si + 1) % 500 == 0:
            print(f"    Site {si+1}/{len(all_Q)}, computed {computed}, skipped {skipped}, {time.time()-t2:.0f}s")

    print(f"  Computed {computed}, skipped {skipped} in {time.time()-t2:.0f}s")

    # Free memory
    del all_Q
    gc.collect()

    # ── Phase 3: Map back and save ──
    print("\nPhase 3: Updating dataset ...")
    for df_idx, metrics in results.items():
        for col, val in metrics.items():
            df.at[df_idx, col] = val

    df.to_csv(CSV_PATH, index=False)

    print(f"\nFinal dataset:")
    for et in df['event_type'].unique():
        sub = df[df['event_type'] == et]
        valid = sub['dyn_BFI'].notna().sum()
        print(f"  {et}: {len(sub)} events, {valid} with dyn_BFI ({100*valid/len(sub):.1f}%)")

    print(f"\nTotal time: {time.time()-t0:.0f}s ({(time.time()-t0)/60:.1f} min)")


if __name__ == '__main__':
    main()
