#!/usr/bin/env python3
"""
Step 2: Recovery Metrics & Atlas (FIXED VERSION)
=================================================
Reads the matched SPEI-SSI pairs from Step 1 and the monthly time series,
then computes recovery metrics with critical fixes:

FIXES APPLIED:
  - Fix 1: Recovery search window extended to 36 months (was 6)
  - Fix 2: Half-life computation uses true half-life (peak/2) not quarter-life
  - Fix 3: Added antecedent_ssi, spei_term_month, and seasonal encodings
  - Fix 4: Also loads SPEI timeseries and computes post-drought rebound metrics

Generates 4 focused publication figures:
  - Fig 1: CONUS Recovery Atlas (4-panel map)
  - Fig 2: Timescale Comparison (histograms)
  - Fig 3: Aridity × BFI × Recovery interaction
  - Fig 4: Recovery Typology (Buffered vs Independent)

Outputs (in this folder):
  drought_matched_pairs_with_recovery.csv
  drought_basin_recovery_metrics.csv
  figures/Fig1–Fig4 (.png + .pdf)
"""

import os
import gc
import warnings
import zipfile
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── paths ──
STEP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CORE_ROOT = STEP_DIR.parent  # Fix_core/
DATA_ROOT = CORE_ROOT.parent  # EGU Earth Future/
STEP1_DIR = CORE_ROOT / 'Step1_Event_Matching'

MATCHED_CSV = STEP1_DIR / 'drought_matched_pairs.csv'
BUFFERED_CSV = STEP1_DIR / 'drought_buffered_events.csv'
INDEPENDENT_CSV = STEP1_DIR / 'drought_independent_events.csv'
TIMESERIES = DATA_ROOT / 'tier1_3606_SPEI_SSI_timeseries.csv'
GAGESII_ZIP = DATA_ROOT / 'basinchar_and_report_sept_2011' / 'spreadsheets-in-csv-format.zip'
GAGESII_CSV = DATA_ROOT / 'gagesii_csv'

FIG_DIR = STEP_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

RECOVERY_THRESH = -0.5   # SSI threshold for "recovery"
RECOVERY_WINDOW_MONTHS = 36  # FIX 1: Extended from 6 to 36


def ts_print(msg):
    """Print timestamped message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def main():
    """Main execution function (callable from run_all.py)."""

    # ═══════════════════════════════════════════════════════════════
    # PART A — COMPUTE RECOVERY METRICS
    # ═══════════════════════════════════════════════════════════════

    ts_print("═══ STEP 2: RECOVERY ATLAS (FIXED) ═══")

    ts_print("Loading matched pairs from Step 1 …")
    pairs = pd.read_csv(MATCHED_CSV, dtype={'GAGE_ID': str})
    ts_print(f"  {len(pairs):,} pairs")

    ts_print("Loading SPEI and SSI time series …")
    ts_df = pd.read_csv(TIMESERIES, dtype={'GAGE_ID': str})
    ts_df['date'] = pd.to_datetime(ts_df['date'])
    ts_df = ts_df.sort_values(['GAGE_ID', 'date']).reset_index(drop=True)

    # Build gauge lookup with BOTH SSI and SPEI timeseries
    gauge_lookup = {}
    for gid, gdf in ts_df.groupby('GAGE_ID'):
        gdf = gdf.sort_values('date')
        gauge_lookup[gid] = {
            'dates': gdf['date'].values,
            'SSI_3': gdf['SSI_3'].values,
            'SSI_6': gdf['SSI_6'].values,
            'SPEI_3': gdf['SPEI_3'].values,
            'SPEI_6': gdf['SPEI_6'].values,
        }
    del ts_df
    gc.collect()
    ts_print(f"  Indexed {len(gauge_lookup)} gauges")

    # Initialize arrays for new metrics
    half_lives = np.zeros(len(pairs), dtype=int)
    half_life_censored = np.zeros(len(pairs), dtype=bool)
    recovery_lags = np.full(len(pairs), np.nan)
    recovery_censored = np.zeros(len(pairs), dtype=bool)
    antecedent_ssi_vals = np.full(len(pairs), np.nan)
    spei_term_months = np.zeros(len(pairs), dtype=int)
    sin_term_months = np.full(len(pairs), np.nan)
    cos_term_months = np.full(len(pairs), np.nan)
    spei_rebound_3mo = np.full(len(pairs), np.nan)
    spei_rebound_6mo = np.full(len(pairs), np.nan)
    post_drought_precip = np.full(len(pairs), np.nan)

    pairs = pairs.reset_index(drop=True)

    ts_print("Computing recovery metrics …")
    for i in range(len(pairs)):
        if (i + 1) % 15000 == 0:
            ts_print(f"  {i+1}/{len(pairs)} …")

        row = pairs.iloc[i]
        gid = row['GAGE_ID']
        gdata = gauge_lookup.get(gid)
        if gdata is None:
            continue

        ts_val = str(int(row['timescale']))
        ssi_col = f'SSI_{ts_val}'
        spei_col = f'SPEI_{ts_val}'
        ssi_vals = gdata[ssi_col]
        spei_vals = gdata[spei_col]
        dates = gdata['dates']

        # Parse dates
        try:
            ssi_onset_dt = np.datetime64(pd.to_datetime(row['ssi_onset']))
            ssi_term_dt = np.datetime64(pd.to_datetime(row['ssi_termination']))
            spei_term_dt = np.datetime64(pd.to_datetime(row['spei_termination']))
        except Exception:
            continue

        # Find indices
        ssi_onset_idx = np.searchsorted(dates, ssi_onset_dt, side='left')
        ssi_term_idx = np.searchsorted(dates, ssi_term_dt, side='right') - 1
        spei_term_idx = np.searchsorted(dates, spei_term_dt, side='right') - 1

        if ssi_onset_idx >= len(ssi_vals) or ssi_term_idx >= len(ssi_vals):
            continue

        # ─────────────────────────────────────────────────────────────
        # FIX 1: Recovery lag (36-month window from SPEI termination)
        # ─────────────────────────────────────────────────────────────
        # FIX: Start from spei_term_idx, not ssi_onset_idx
        # FIX: Extend window to 36 months
        if spei_term_idx < len(ssi_vals):
            search_end = min(spei_term_idx + RECOVERY_WINDOW_MONTHS, len(ssi_vals))
            for k in range(spei_term_idx, search_end):
                v = ssi_vals[k]
                if not np.isnan(v) and v > RECOVERY_THRESH:
                    recovery_lags[i] = k - spei_term_idx
                    recovery_censored[i] = False
                    break
            # If no recovery found, mark as censored and set to window length
            if np.isnan(recovery_lags[i]):
                recovery_lags[i] = search_end - spei_term_idx
                recovery_censored[i] = True

        # ─────────────────────────────────────────────────────────────
        # FIX 2: Half-life computation (true half-life = peak/2.0)
        # ─────────────────────────────────────────────────────────────
        # Find peak (minimum SSI) during SSI event
        peak_val = np.inf
        peak_idx = ssi_onset_idx
        for k in range(ssi_onset_idx, min(ssi_term_idx + 1, len(ssi_vals))):
            v = ssi_vals[k]
            if not np.isnan(v) and v < peak_val:
                peak_val = v
                peak_idx = k

        if not np.isinf(peak_val):
            # FIX 2: True half-life target is peak_val / 2.0 (halfway to zero)
            half_life_target = peak_val / 2.0

            # Scan from peak forward up to 36-month window
            hl = None
            search_end_hl = min(spei_term_idx + RECOVERY_WINDOW_MONTHS, len(ssi_vals))
            for k in range(peak_idx + 1, search_end_hl):
                v = ssi_vals[k]
                if not np.isnan(v) and v >= half_life_target:
                    hl = k - peak_idx
                    half_life_censored[i] = False
                    break

            if hl is None:
                hl = search_end_hl - peak_idx  # censored
                half_life_censored[i] = True

            half_lives[i] = max(hl, 0)

        # ─────────────────────────────────────────────────────────────
        # FIX 3: Antecedent SSI and seasonal encoding
        # ─────────────────────────────────────────────────────────────
        if spei_term_idx < len(ssi_vals):
            v = ssi_vals[spei_term_idx]
            if not np.isnan(v):
                antecedent_ssi_vals[i] = v

        # Extract month from spei_termination date
        spei_term_date = pd.to_datetime(row['spei_termination'])
        month = spei_term_date.month
        spei_term_months[i] = month
        sin_term_months[i] = np.sin(2 * np.pi * month / 12.0)
        cos_term_months[i] = np.cos(2 * np.pi * month / 12.0)

        # ─────────────────────────────────────────────────────────────
        # FIX 4: SPEI rebound metrics (post-drought SPEI signal)
        # ─────────────────────────────────────────────────────────────
        if spei_term_idx < len(spei_vals):
            # 3-month rebound
            rebound_3_end = min(spei_term_idx + 3, len(spei_vals))
            rebound_3_vals = spei_vals[spei_term_idx:rebound_3_end]
            rebound_3_valid = rebound_3_vals[~np.isnan(rebound_3_vals)]
            if len(rebound_3_valid) > 0:
                spei_rebound_3mo[i] = np.mean(rebound_3_valid)

            # 6-month rebound
            rebound_6_end = min(spei_term_idx + 6, len(spei_vals))
            rebound_6_vals = spei_vals[spei_term_idx:rebound_6_end]
            rebound_6_valid = rebound_6_vals[~np.isnan(rebound_6_vals)]
            if len(rebound_6_valid) > 0:
                spei_rebound_6mo[i] = np.mean(rebound_6_valid)

            # Cumulative SPEI in 6 months after termination (post-drought precipitation signal)
            cum_spei = np.nansum(spei_vals[spei_term_idx:rebound_6_end])
            if not np.isnan(cum_spei):
                post_drought_precip[i] = cum_spei

    # Add new columns to pairs dataframe
    pairs['recovery_lag_months'] = recovery_lags
    pairs['recovery_censored'] = recovery_censored
    pairs['recovery_half_life_months'] = half_lives
    pairs['half_life_censored'] = half_life_censored
    pairs['antecedent_ssi'] = antecedent_ssi_vals
    pairs['spei_term_month'] = spei_term_months
    pairs['sin_term_month'] = sin_term_months
    pairs['cos_term_month'] = cos_term_months
    pairs['spei_rebound_3mo'] = spei_rebound_3mo
    pairs['spei_rebound_6mo'] = spei_rebound_6mo
    pairs['post_drought_precip_signal'] = post_drought_precip

    # Rename old column for clarity
    if 'ssi_false_rec' in pairs.columns:
        pairs['ssi_had_false_recovery'] = (pairs['ssi_false_rec'] > 0).astype(int)

    ts_print(f"  Half-life: mean={half_lives.mean():.1f}, median={np.median(half_lives):.1f}")
    ts_print(f"  Recovery lag valid: {np.isfinite(recovery_lags).sum():,}")
    ts_print(f"  Recovery censored: {recovery_censored.sum():,}")
    ts_print(f"  Half-life censored: {half_life_censored.sum():,}")

    del gauge_lookup
    gc.collect()

    # Save master dataset
    out_path = STEP_DIR / 'drought_matched_pairs_with_recovery.csv'
    pairs.to_csv(out_path, index=False)
    ts_print(f"  Saved {out_path.name} ({len(pairs):,} rows)")

    # ═══════════════════════════════════════════════════════════════
    # PART B — BASIN-LEVEL AGGREGATION
    # ═══════════════════════════════════════════════════════════════

    ts_print("\nAggregating per-basin recovery metrics …")
    basin_list = []
    for ts_val in [3, 6]:
        sub = pairs[pairs['timescale'] == ts_val]
        agg = sub.groupby('GAGE_ID').agg(
            n_events=('recovery_lag_months', 'count'),
            prop_lag_median=('propagation_lag_months', 'median'),
            prop_lag_mean=('propagation_lag_months', 'mean'),
            rec_lag_median=('recovery_lag_months', 'median'),
            rec_lag_mean=('recovery_lag_months', 'mean'),
            half_life_median=('recovery_half_life_months', 'median'),
            half_life_mean=('recovery_half_life_months', 'mean'),
            false_rec_prob=('ssi_had_false_recovery', 'mean'),
            spei_sev_mean=('spei_severity', 'mean'),
            ssi_sev_mean=('ssi_severity', 'mean'),
            antecedent_ssi_median=('antecedent_ssi', 'median'),
            spei_rebound_3mo_median=('spei_rebound_3mo', 'median'),
            spei_rebound_6mo_median=('spei_rebound_6mo', 'median'),
        ).reset_index()
        agg['timescale'] = ts_val
        basin_list.append(agg)

    basin_df = pd.concat(basin_list, ignore_index=True)
    basin_df.to_csv(STEP_DIR / 'drought_basin_recovery_metrics.csv', index=False)
    ts_print(f"  Saved drought_basin_recovery_metrics.csv ({len(basin_df)} rows)")

    # ═══════════════════════════════════════════════════════════════
    # PART C — PUBLICATION FIGURES (4 figures)
    # ═══════════════════════════════════════════════════════════════

    ts_print("\n═══ FIGURES ═══")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats as sp_stats

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 11,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
        'savefig.facecolor': 'white', 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': 0.6,
    })

    def _save(fig, name):
        """Save figure to PNG and PDF."""
        for ext in ['png', 'pdf']:
            fig.savefig(FIG_DIR / f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        ts_print(f"  Saved {name}")

    def _make_conus_ax(fig, pos):
        """Create CONUS axes with offline Natural Earth boundaries."""
        import geopandas as _gpd
        ax = fig.add_subplot(pos)
        ax.set_xlim(-125, -66)
        ax.set_ylim(24, 50)
        ax.set_aspect('equal')
        # Try portable approaches for state/country boundaries
        try:
            world = _gpd.read_file(_gpd.datasets.get_path('naturalearth_lowres'))
            usa = world[world['name'] == 'United States of America']
            usa.boundary.plot(ax=ax, color='0.4', linewidth=0.5)
            # Add neighbors for context
            neighbors = world[world['name'].isin(['Canada', 'Mexico'])]
            neighbors.boundary.plot(ax=ax, color='0.6', linewidth=0.3)
        except Exception:
            ax.plot([-125, -66, -66, -125, -125], [24, 24, 50, 50, 24], 'k-', lw=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    # Load GAGES-II coords for maps
    def _read_gages_csv(fname, cols=None):
        """Read GAGES-II CSV with standard transformations."""
        fpath = GAGESII_CSV / fname
        if not fpath.is_file():
            return None
        df = pd.read_csv(fpath, encoding='latin1', dtype={'STAID': str})
        df['GAGE_ID'] = df['STAID'].astype(str).str.lstrip('0')
        df.drop(columns=['STAID'], inplace=True, errors='ignore')
        if cols:
            cols = [c for c in cols if c in df.columns]
            return df[['GAGE_ID'] + cols]
        return df

    if not GAGESII_CSV.is_dir() and GAGESII_ZIP.is_file():
        ts_print("  Extracting GAGES-II CSVs …")
        with zipfile.ZipFile(GAGESII_ZIP, 'r') as zf:
            zf.extractall(GAGESII_CSV)

    coords = None
    classif = None
    if GAGESII_CSV.is_dir():
        coords = _read_gages_csv('conterm_basinid.txt', ['LAT_GAGE', 'LNG_GAGE'])
        classif = _read_gages_csv('conterm_bas_classif.txt', ['AGGECOREGION'])
        if classif is not None and coords is not None:
            coords = coords.merge(classif, on='GAGE_ID', how='left')

    # Merge coords into basin_df for TS=6
    b6 = basin_df[basin_df['timescale'] == 6].copy()
    if coords is not None:
        b6 = b6.merge(coords, on='GAGE_ID', how='left')

    # Get BFI for Fig 3
    hydro = _read_gages_csv('conterm_hydro.txt', ['BFI_AVE']) if GAGESII_CSV.is_dir() else None
    if hydro is not None:
        b6 = b6.merge(hydro, on='GAGE_ID', how='left')

    # Get climate/aridity for Fig 3
    climate = _read_gages_csv('conterm_climate.txt', ['PPTAVG_BASIN', 'PET']) if GAGESII_CSV.is_dir() else None
    if climate is not None:
        climate['ARIDITY_INDEX'] = climate['PET'] / climate['PPTAVG_BASIN'].replace(0, np.nan)
        b6 = b6.merge(climate[['GAGE_ID', 'ARIDITY_INDEX']], on='GAGE_ID', how='left')

    # ── Fig 1: Multi-panel CONUS Recovery Atlas (4-panel) ──
    ts_print("Fig 1: CONUS Recovery Atlas …")
    valid = b6.dropna(subset=['LNG_GAGE', 'LAT_GAGE']).copy()
    valid = valid[(valid['LNG_GAGE'] > -130) & (valid['LNG_GAGE'] < -65) &
                  (valid['LAT_GAGE'] > 24) & (valid['LAT_GAGE'] < 50)]

    metrics = [
        ('prop_lag_median', 'Median propagation lag (mo)', 'RdYlBu_r', 0, 6),
        ('rec_lag_median', 'Median recovery lag (mo)', 'RdYlBu_r', 0, 15),
        ('half_life_median', 'Median recovery half-life (mo)', 'RdYlBu_r', 0, 8),
        ('false_rec_prob', 'False recovery probability', 'Reds', 0, 0.5),
    ]

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    for idx, (col, title, cmap, vmin, vmax) in enumerate(metrics):
        ax = fig.add_subplot(gs[idx // 2, idx % 2])
        v = valid.dropna(subset=[col])
        sc = ax.scatter(v['LNG_GAGE'], v['LAT_GAGE'], c=v[col].clip(vmin, vmax),
                        cmap=cmap, s=5, alpha=0.7, vmin=vmin, vmax=vmax,
                        edgecolors='none', rasterized=True)
        ax.set_xlim(-128, -65)
        ax.set_ylim(24, 50)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'({chr(97+idx)}) {title}', fontweight='bold')
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(f'Hydrological drought recovery atlas — CONUS ({len(valid):,} basins, SPEI-6/SSI-6)',
                 fontsize=13, fontweight='bold', y=0.995)
    _save(fig, 'Fig1_CONUS_Recovery_Atlas')

    # ── Fig 2: Timescale Comparison (propagation and recovery lags) ──
    ts_print("Fig 2: Timescale Comparison …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ai, (metric, label) in enumerate([
        ('propagation_lag_months', 'Propagation lag (months)'),
        ('recovery_lag_months', 'Recovery lag (months)')
    ]):
        ax = axes[ai]
        for ts_val, col, lab in [(3, '#2166ac', 'SPEI-3/SSI-3'), (6, '#b2182b', 'SPEI-6/SSI-6')]:
            vals = pairs[pairs['timescale'] == ts_val][metric].dropna()
            ax.hist(vals.clip(-5, 40), bins=45, alpha=0.55, color=col,
                    label=f'{lab} (n={len(vals):,})', density=True, edgecolor='none')
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'({chr(97+ai)}) {label}', fontweight='bold')
        ax.legend(fontsize=9, loc='upper right')
        ax.grid(axis='y', alpha=0.2)
        ax.set_xlim(-5, 40)

    fig.suptitle('Timescale comparison: SPEI-3/SSI-3 vs SPEI-6/SSI-6',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'Fig2_Timescale_Comparison')

    # ── Fig 3: Aridity × BFI × Recovery interaction ──
    ts_print("Fig 3: Aridity × BFI × Recovery …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if 'ARIDITY_INDEX' in b6.columns and 'BFI_AVE' in b6.columns:
        v3 = b6[['ARIDITY_INDEX', 'BFI_AVE', 'half_life_median']].dropna()

        # Panel (a): 3D-like scatter
        ax = axes[0]
        sc3 = ax.scatter(v3['ARIDITY_INDEX'].clip(0, 8), v3['half_life_median'],
                         c=v3['BFI_AVE'], cmap='RdYlBu', s=12, alpha=0.6,
                         edgecolors='none', rasterized=True, vmin=0, vmax=100)
        cbar = plt.colorbar(sc3, ax=ax, shrink=0.8)
        cbar.set_label('Baseflow Index (%)', fontsize=9)
        ax.set_xlabel('Aridity Index (PET/P)', fontsize=10)
        ax.set_ylabel('Median recovery half-life (months)', fontsize=10)
        ax.set_title('(a) Aridity × BFI × Recovery half-life', fontweight='bold')
        ax.grid(axis='both', alpha=0.15)

        # Panel (b): Heatmap of quartile interaction
        ax = axes[1]
        v3['ai_q'] = pd.qcut(v3['ARIDITY_INDEX'], 4,
                             labels=['Q1\n(humid)', 'Q2', 'Q3', 'Q4\n(arid)'],
                             duplicates='drop')
        v3['bfi_q'] = pd.qcut(v3['BFI_AVE'], 3,
                              labels=['Low\nBFI', 'Mid\nBFI', 'High\nBFI'],
                              duplicates='drop')
        piv = v3.pivot_table('half_life_median', index='bfi_q', columns='ai_q', aggfunc='median')

        im = ax.imshow(piv.values, cmap='RdYlBu_r', aspect='auto', vmin=2, vmax=8)
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, fontsize=9)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=9)

        # Annotate heatmap
        for ii in range(piv.shape[0]):
            for jj in range(piv.shape[1]):
                vv = piv.values[ii, jj]
                if np.isfinite(vv):
                    ax.text(jj, ii, f'{vv:.1f}', ha='center', va='center', fontsize=10,
                            fontweight='bold', color='white' if vv > 5 else 'black')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Median half-life (mo)', fontsize=9)
        ax.set_title('(b) Aridity × BFI quartile interaction', fontweight='bold')

    fig.suptitle('Recovery characteristics: Aridity and Baseflow Index',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    _save(fig, 'Fig3_Aridity_BFI_Recovery')

    # ── Fig 4: Recovery Typology (Buffered vs Independent) ──
    ts_print("Fig 4: Recovery Typology …")

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1.3, 0.9], wspace=0.3)

    # Load buffered and independent events
    buf_events = None
    ind_events = None

    if BUFFERED_CSV.is_file():
        buf_events = pd.read_csv(BUFFERED_CSV, dtype={'GAGE_ID': str})
    if INDEPENDENT_CSV.is_file():
        ind_events = pd.read_csv(INDEPENDENT_CSV, dtype={'GAGE_ID': str})

    # Panel (a): Histogram of buffered SPEI severity + independent SSI duration
    ax = fig.add_subplot(gs[0])
    if buf_events is not None:
        for ts_val, col, lab in [(3, '#2166ac', 'SPEI-3'), (6, '#b2182b', 'SPEI-6')]:
            bs = buf_events[buf_events['timescale'] == ts_val]['severity'].dropna()
            if len(bs) > 0:
                ax.hist(bs.clip(-30, 0), bins=30, alpha=0.55, color=col,
                        label=f'Buffered {lab} (n={len(bs):,})', density=True, edgecolor='none')
    ax.set_xlabel('Cumulative SPEI severity', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('(a) Buffered SPEI events', fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=0.2)

    # Panel (b): CONUS map of buffered vs independent distribution
    ax = fig.add_subplot(gs[1])
    ax.set_xlim(-128, -65)
    ax.set_ylim(24, 50)
    ax.set_aspect('equal')

    # Build typology per basin
    typology_basins = []
    if buf_events is not None and ind_events is not None:
        buf_basins = buf_events.groupby('GAGE_ID').size()
        ind_basins = ind_events.groupby('GAGE_ID').size()
        all_basins = set(buf_basins.index) | set(ind_basins.index)

        for gid in all_basins:
            buf_count = buf_basins.get(gid, 0)
            ind_count = ind_basins.get(gid, 0)

            if buf_count > ind_count * 2:
                typology = 'buffered'
                color = '#2166ac'
            elif ind_count > buf_count * 2:
                typology = 'independent'
                color = '#b2182b'
            else:
                typology = 'mixed'
                color = '#b4469a'

            typology_basins.append({
                'GAGE_ID': gid,
                'typology': typology,
                'color': color,
                'buf_count': buf_count,
                'ind_count': ind_count,
            })

    # Merge typology with coordinates
    if typology_basins and coords is not None:
        typology_df = pd.DataFrame(typology_basins)
        typology_df = typology_df.merge(coords, on='GAGE_ID', how='left')
        typology_df = typology_df.dropna(subset=['LNG_GAGE', 'LAT_GAGE'])
        typology_df = typology_df[(typology_df['LNG_GAGE'] > -130) & (typology_df['LNG_GAGE'] < -65) &
                                  (typology_df['LAT_GAGE'] > 24) & (typology_df['LAT_GAGE'] < 50)]

        for typ, col in [('buffered', '#2166ac'), ('independent', '#b2182b'), ('mixed', '#b4469a')]:
            subset = typology_df[typology_df['typology'] == typ]
            if len(subset) > 0:
                ax.scatter(subset['LNG_GAGE'], subset['LAT_GAGE'], c=col, s=6, alpha=0.6,
                          label=f'{typ.capitalize()} (n={len(subset)})', edgecolors='none',
                          rasterized=True)

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title('(b) Spatial distribution of recovery typology', fontweight='bold')
    ax.legend(fontsize=8, loc='lower left')

    # Panel (c): Summary bar chart
    ax = fig.add_subplot(gs[2])
    if typology_basins:
        typology_counts = pd.Series([t['typology'] for t in typology_basins]).value_counts()
        colors_bar = {'buffered': '#2166ac', 'independent': '#b2182b', 'mixed': '#b4469a'}
        ordered_types = ['buffered', 'independent', 'mixed']
        ordered_counts = [typology_counts.get(t, 0) for t in ordered_types]
        ordered_colors = [colors_bar[t] for t in ordered_types if typology_counts.get(t, 0) > 0]
        ordered_labels = [t for t in ordered_types if typology_counts.get(t, 0) > 0]
        ordered_counts = [c for c in ordered_counts if c > 0]

        bars = ax.bar(ordered_labels, ordered_counts, color=ordered_colors, edgecolor='black', linewidth=0.8)
        ax.set_ylabel('Number of basins', fontsize=10)
        ax.set_title('(c) Typology counts', fontweight='bold')
        ax.grid(axis='y', alpha=0.2)

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.suptitle('Recovery typology: Buffered vs Independent drought behavior',
                 fontsize=12, fontweight='bold')
    _save(fig, 'Fig4_Recovery_Typology')

    ts_print(f"\nAll outputs saved to {STEP_DIR}/")
    ts_print("STEP 2: RECOVERY ATLAS (FIXED) — COMPLETE")


if __name__ == '__main__':
    main()
