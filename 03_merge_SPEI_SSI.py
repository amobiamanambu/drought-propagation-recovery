"""
Merge SPEI and SSI into a Single Tier 1 Time-Series File
================================================================================
Reads the LONG-format SPEI output from Script 01 and the WIDE-format SSI
output from Script 02, pivots SPEI to wide, and merges on (basin_id, date)
for non-arid Tier 1 basins.

Inputs (relative to parent directory):
    - basin_drought_analysis/basin_spei.csv        (LONG: basin_id, GAGE_ID, date, timescale, SPEI)
    - basin_drought_indices/basin_SSI_timeseries.csv (WIDE: basin_id, GAGE_ID, date, SSI_3, SSI_6)
    - quality_control/tier1_non_arid_basins.csv     (basin list from Script 02)

Output:
    - Data/tier1_3606_SPEI_SSI_timeseries.csv
      Columns: basin_id, GAGE_ID, date, SPEI_3, SPEI_6, SSI_3, SSI_6

Usage:
    cd Code/
    python3 03_merge_SPEI_SSI.py
================================================================================
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("MERGE SPEI + SSI INTO TIER 1 TIME-SERIES FILE")
print("=" * 80)

# ============================================================================
# PATH CONFIGURATION
# ============================================================================
SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR   = SCRIPT_DIR.parent

SPEI_FILE   = BASE_DIR / "basin_drought_analysis" / "basin_spei.csv"
SSI_FILE    = BASE_DIR / "basin_drought_indices"   / "basin_SSI_timeseries.csv"
TIER1_FILE  = BASE_DIR / "quality_control"         / "tier1_non_arid_basins.csv"

DATA_DIR    = BASE_DIR / "Data"
OUTPUT_FILE = DATA_DIR / "tier1_3606_SPEI_SSI_timeseries.csv"

TIMESCALES = [3, 6]
CHUNKSIZE  = 100_000

# ============================================================================
# MAIN
# ============================================================================

def main():
    start = datetime.now()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load non-arid Tier 1 basin list
    # ------------------------------------------------------------------
    print("\n1. Loading Tier 1 non-arid basin list...")
    tier1_df = pd.read_csv(TIER1_FILE)
    tier1_ids = set(tier1_df['basin_id'])
    print(f"   Tier 1 non-arid basins: {len(tier1_ids):,}")

    # ------------------------------------------------------------------
    # 2. Index SPEI (LONG format) by basin — filter to timescales 3 & 6
    # ------------------------------------------------------------------
    print("\n2. Indexing SPEI data (timescales 3, 6)...")
    spei_by_basin = {}
    for chunk_num, chunk in enumerate(pd.read_csv(SPEI_FILE, chunksize=CHUNKSIZE), 1):
        chunk = chunk[chunk['timescale'].isin(TIMESCALES)].copy()
        for bid in chunk['basin_id'].unique():
            if bid in tier1_ids:
                spei_by_basin.setdefault(bid, []).append(
                    chunk[chunk['basin_id'] == bid]
                )
        if chunk_num % 10 == 0:
            print(f"   SPEI chunk {chunk_num} | basins indexed: {len(spei_by_basin):,}")

    print(f"   SPEI indexed for {len(spei_by_basin):,} basins")

    # ------------------------------------------------------------------
    # 3. Index SSI (WIDE format) by basin
    # ------------------------------------------------------------------
    print("\n3. Indexing SSI data...")
    ssi_by_basin = {}
    for chunk_num, chunk in enumerate(pd.read_csv(SSI_FILE, chunksize=CHUNKSIZE), 1):
        for bid in chunk['basin_id'].unique():
            if bid in tier1_ids:
                ssi_by_basin.setdefault(bid, []).append(
                    chunk[chunk['basin_id'] == bid]
                )
        if chunk_num % 10 == 0:
            print(f"   SSI chunk {chunk_num} | basins indexed: {len(ssi_by_basin):,}")

    print(f"   SSI indexed for {len(ssi_by_basin):,} basins")

    # ------------------------------------------------------------------
    # 4. Merge basin-by-basin and write incrementally
    # ------------------------------------------------------------------
    print("\n4. Merging basin-by-basin...")
    first_write = True
    records_total = 0

    all_basins = sorted(tier1_ids)
    for bi, basin_id in enumerate(all_basins):
        if (bi + 1) % 500 == 0 or bi == 0:
            print(f"   [{bi+1}/{len(all_basins)}] records written: {records_total:,}")

        parts = []

        # SPEI: pivot LONG -> WIDE
        if basin_id in spei_by_basin:
            spei_long = pd.concat(spei_by_basin[basin_id], ignore_index=True)
            spei_wide = spei_long.pivot_table(
                index=['basin_id', 'date'],
                columns='timescale',
                values='SPEI',
                aggfunc='first'
            ).reset_index()
            spei_wide.columns = ['basin_id', 'date'] + \
                [f'SPEI_{int(c)}' for c in spei_wide.columns[2:]]

            # Carry GAGE_ID if present
            if 'GAGE_ID' in spei_long.columns:
                gage_map = spei_long.drop_duplicates('basin_id')[['basin_id', 'GAGE_ID']]
                spei_wide = spei_wide.merge(gage_map, on='basin_id', how='left')

            parts.append(spei_wide)

        # SSI (already WIDE)
        if basin_id in ssi_by_basin:
            ssi_basin = pd.concat(ssi_by_basin[basin_id], ignore_index=True)
            keep = [c for c in ['basin_id', 'date', 'GAGE_ID', 'SSI_3', 'SSI_6']
                    if c in ssi_basin.columns]
            parts.append(ssi_basin[keep])

        if len(parts) == 0:
            continue

        merged = parts[0]
        for p in parts[1:]:
            merged = merged.merge(p, on=['basin_id', 'date'], how='outer',
                                  suffixes=('', '_dup'))

        # Resolve duplicate GAGE_ID
        if 'GAGE_ID_dup' in merged.columns:
            merged['GAGE_ID'] = merged['GAGE_ID'].fillna(merged['GAGE_ID_dup'])
            merged.drop('GAGE_ID_dup', axis=1, inplace=True)

        merged['date'] = pd.to_datetime(merged['date'])
        merged = merged.sort_values('date').reset_index(drop=True)

        col_order = ['basin_id', 'GAGE_ID', 'date',
                     'SPEI_3', 'SPEI_6', 'SSI_3', 'SSI_6']
        col_order = [c for c in col_order if c in merged.columns]
        merged = merged[col_order]

        if first_write:
            merged.to_csv(OUTPUT_FILE, index=False, mode='w')
            first_write = False
        else:
            merged.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)

        records_total += len(merged)

    # ------------------------------------------------------------------
    # 5. Verify
    # ------------------------------------------------------------------
    print("\n5. Verifying output...")
    if OUTPUT_FILE.exists():
        sample = pd.read_csv(OUTPUT_FILE, nrows=1000)
        size_mb = OUTPUT_FILE.stat().st_size / 1024**2

        # Count total rows
        total_rows = 0
        for chunk in pd.read_csv(OUTPUT_FILE, chunksize=CHUNKSIZE):
            total_rows += len(chunk)

        print(f"   File:    {OUTPUT_FILE}")
        print(f"   Size:    {size_mb:.1f} MB")
        print(f"   Records: {total_rows:,}")
        print(f"   Basins:  {sample['basin_id'].nunique()} (in first 1k rows)")
        print(f"   Columns: {list(sample.columns)}")

        print(f"\n   Completeness (sample):")
        for col in ['SPEI_3', 'SPEI_6', 'SSI_3', 'SSI_6']:
            if col in sample.columns:
                pct = sample[col].notna().mean() * 100
                print(f"     {col}: {pct:.1f}%")
    else:
        print("   WARNING: Output file not found!")

    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n{'='*80}")
    print("MERGE COMPLETE")
    print(f"{'='*80}")
    print(f"  Output:  {OUTPUT_FILE}")
    print(f"  Records: {records_total:,}")
    print(f"  Time:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 80)


if __name__ == "__main__":
    main()
