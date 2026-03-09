#!/usr/bin/env python3
"""
Step 4: Event-Level Hierarchical Attribution of Hydrological Drought Recovery
==============================================================================
COMPLETELY REWRITTEN: Event-level (not basin median) analysis with 5-group Shapley partitioning.

Reads matched pairs from Step 2 (event-level recovery lags), memory metrics from Step 3,
and GAGES-II attributes. Performs:
  - Event-level hierarchical regression (5 groups: Climate_Rebound, Climate_Context,
    Catchment_Memory, Static_Landscape, Event_Chars)
  - Shapley-value variance partitioning (120 permutations of 5 groups)
  - Random Forest with nonlinear interaction detection
  - Ridge regression operational model with interaction terms
  - Recovery regime classification (basin-level)
  - 5 publication figures (Hierarchical R², Shapley, Regime Map, Memory Signature, Operational)

Timescales: SPEI-3 and SPEI-6.
Output: attribution_results/ with CSVs, text summary, and 5 figures (PNG + PDF).
"""

import os, sys, gc, time, warnings, zipfile
from pathlib import Path
from itertools import permutations
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ── Memory helpers ──
def _shrink(df):
    """Downcast numeric columns to save memory."""
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= -32768 and df[col].max() <= 32767:
            df[col] = df[col].astype(np.int16)
        elif df[col].min() >= 0 and df[col].max() <= 65535:
            df[col] = df[col].astype(np.uint16)
        else:
            df[col] = df[col].astype(np.int32)
    return df

def ts_print(msg):
    """Timestamp print."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# ── Paths ──
STEP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
CORE_ROOT = STEP_DIR.parent
DATA_ROOT = CORE_ROOT.parent
STEP2_DIR = CORE_ROOT / 'Step2_Recovery_Atlas'
STEP3_DIR = CORE_ROOT / 'Step3_Memory_and_Covariates'
OUT = STEP_DIR / 'attribution_results'
OUT.mkdir(exist_ok=True)

# Primary: dynamic matched pairs from Step 3 (has dyn_BFI, dyn_recession_k_days, etc.)
DYNAMIC_CSV    = STEP3_DIR / 'dynamic_memory_matched_pairs.csv'
# Fallback: Step 2 output + static memory
MATCHED_CSV    = STEP2_DIR / 'drought_matched_pairs_with_recovery.csv'
MEMORY_CSV     = STEP3_DIR / 'catchment_memory_metrics.csv'
GAGESII_ZIP    = DATA_ROOT / 'basinchar_and_report_sept_2011' / 'spreadsheets-in-csv-format.zip'
GAGESII_CSV    = DATA_ROOT / 'gagesii_csv'

# Group definitions (5 groups, 120 permutations)
# NOTE: Catchment_Memory now uses DYNAMIC (event-paired) metrics from Step 3
GROUP_DEFS = {
    'Climate_Rebound': ['spei_rebound_3mo', 'spei_rebound_6mo', 'post_drought_precip_signal'],
    'Climate_Context': ['ARIDITY_INDEX', 'PET', 'PPTAVG_BASIN', 'SNOW_PCT_PRECIP'],
    'Catchment_Memory': ['dyn_BFI', 'dyn_recession_k_days', 'dyn_cv_Q', 'dyn_zero_flow_fraction'],
    'Static_Landscape': ['PERMAVE', 'FORESTNLCD06', 'ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'WTDEPAVE', 'SANDAVE'],
    'Event_Chars': ['spei_severity', 'spei_duration', 'propagation_lag_months',
                    'sin_term_month', 'cos_term_month', 'antecedent_ssi'],
}

GROUP_COLORS = {
    'Climate_Rebound': '#2166ac',
    'Climate_Context': '#74add1',
    'Catchment_Memory': '#b2182b',
    'Static_Landscape': '#1a9850',
    'Event_Chars': '#e08214'
}

TARGET = 'term_to_recovery_lag_months'  # Event-level recovery lag from Step 2

def main():
    """Main analysis pipeline."""

    # ══════════════════════════════════════════════════════════════════
    # PART A – DATA ASSEMBLY (EVENT LEVEL)
    # ══════════════════════════════════════════════════════════════════
    ts_print("═══ PART A: EVENT-LEVEL DATA ASSEMBLY ═══")

    # A1 – Load matched pairs (prefer dynamic CSV from Step 3 if available)
    if DYNAMIC_CSV.exists():
        ts_print("Loading DYNAMIC matched pairs from Step 3 …")
        pairs = pd.read_csv(DYNAMIC_CSV, dtype={'GAGE_ID': str})
        ts_print(f"  Dynamic metrics found: dyn_BFI, dyn_recession_k_days, etc.")
    else:
        ts_print("Loading matched pairs from Step 2 (no dynamic metrics) …")
        pairs = pd.read_csv(MATCHED_CSV, dtype={'GAGE_ID': str})
    pairs = pairs[pairs['timescale'].isin([3, 6])].copy()
    _shrink(pairs)
    ts_print(f"  {len(pairs):,} events, {pairs['GAGE_ID'].nunique()} basins")

    # Verify target column exists
    if TARGET not in pairs.columns:
        ts_print(f"  WARNING: {TARGET} not found. Checking alternatives …")
        if 'recovery_lag_months' in pairs.columns:
            TARGET_USE = 'recovery_lag_months'
            ts_print(f"  Using recovery_lag_months instead")
        else:
            ts_print("  ERROR: No recovery lag column found!")
            return
    else:
        TARGET_USE = TARGET

    # Verify season encoding columns exist
    season_cols = ['sin_term_month', 'cos_term_month']
    for col in season_cols:
        if col not in pairs.columns:
            ts_print(f"  Computing {col} …")
            if 'spei_term_month' in pairs.columns:
                month_vals = pairs['spei_term_month'].values.astype(float)
                if col == 'sin_term_month':
                    pairs[col] = np.sin(2 * np.pi * month_vals / 12)
                else:
                    pairs[col] = np.cos(2 * np.pi * month_vals / 12)

    # A2 – Merge catchment memory (skip if dynamic metrics already present)
    if 'dyn_BFI' in pairs.columns:
        ts_print("Dynamic memory metrics already in pairs — skipping merge")
        ts_print(f"  dyn_BFI available for {pairs['dyn_BFI'].notna().sum():,} events")
    elif MEMORY_CSV.exists():
        ts_print("Merging STATIC catchment memory metrics (fallback) …")
        mem = pd.read_csv(MEMORY_CSV, dtype={'GAGE_ID': str})
        pairs = pairs.merge(mem, on='GAGE_ID', how='left')
        # Rename static to dyn_ for compatibility with GROUP_DEFS
        for old, new in [('BFI', 'dyn_BFI'), ('recession_k_days', 'dyn_recession_k_days'),
                         ('cv_Q', 'dyn_cv_Q'), ('zero_flow_fraction', 'dyn_zero_flow_fraction')]:
            if old in pairs.columns and new not in pairs.columns:
                pairs[new] = pairs[old]
        ts_print(f"  dyn_BFI available for {pairs['dyn_BFI'].notna().sum():,} events")
    else:
        ts_print("  WARNING: No memory metrics found!")

    # A3 – GAGES-II static attributes
    ts_print("Loading GAGES-II attributes …")

    def _read_gages_csv(fname, cols=None):
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

    # Auto-extract CSVs from zip if needed
    if not GAGESII_CSV.is_dir() and GAGESII_ZIP.is_file():
        ts_print("  Extracting CSVs from zip …")
        with zipfile.ZipFile(GAGESII_ZIP, 'r') as zf:
            zf.extractall(GAGESII_CSV)

    if GAGESII_CSV.is_dir():
        attrs = _read_gages_csv('conterm_basinid.txt',
                                 ['DRAIN_SQKM', 'LAT_GAGE', 'LNG_GAGE', 'HUC02'])
        for fname, cols in [
            ('conterm_climate.txt', ['PPTAVG_BASIN', 'T_AVG_BASIN', 'PET',
                                      'SNOW_PCT_PRECIP', 'PRECIP_SEAS_IND', 'RH_BASIN']),
            ('conterm_topo.txt', ['ELEV_MEAN_M_BASIN', 'SLOPE_PCT', 'RRMEAN']),
            ('conterm_soils.txt', ['AWCAVE', 'PERMAVE', 'WTDEPAVE', 'ROCKDEPAVE',
                                    'CLAYAVE', 'SILTAVE', 'SANDAVE']),
            ('conterm_lc06_basin.txt', ['DEVNLCD06', 'FORESTNLCD06', 'CROPSNLCD06',
                                         'SHRUBNLCD06', 'GRASSNLCD06']),
        ]:
            extra = _read_gages_csv(fname, cols)
            if extra is not None:
                attrs = attrs.merge(extra, on='GAGE_ID', how='left')

        # Compute aridity index
        if 'PET' in attrs.columns and 'PPTAVG_BASIN' in attrs.columns:
            attrs['ARIDITY_INDEX'] = attrs['PET'] / attrs['PPTAVG_BASIN'].replace(0, np.nan)

        ts_print(f"  GAGES-II: {len(attrs)} basins × {len(attrs.columns)} cols")
    else:
        ts_print("  WARNING: No GAGES-II data found")
        attrs = pd.DataFrame({'GAGE_ID': pairs['GAGE_ID'].unique()})

    pairs = pairs.merge(attrs, on='GAGE_ID', how='left')
    _shrink(pairs)
    n_geo = pairs['LAT_GAGE'].notna().sum()
    ts_print(f"  Merged: {n_geo:,} events with GAGES-II ({100*n_geo/len(pairs):.1f}%)")

    pairs.to_csv(OUT / 'assembled_event_data.csv', index=False)
    ts_print(f"  Saved {len(pairs):,} event rows")

    # ══════════════════════════════════════════════════════════════════
    # PART B – EVENT-LEVEL HIERARCHICAL REGRESSION + SHAPLEY
    # ══════════════════════════════════════════════════════════════════
    ts_print("\n═══ PART B: EVENT-LEVEL HIERARCHICAL REGRESSION + SHAPLEY ═══")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score, KFold
    from scipy.stats import spearmanr

    hier_results = {}
    shapley_df_list = []
    report_lines = []

    group_order = list(GROUP_DEFS.keys())

    for ts_val in [3, 6]:
        ts_print(f"\n── Timescale {ts_val} ──")
        edf = pairs[pairs['timescale'] == ts_val].copy()
        ts_print(f"  {len(edf):,} events at SPEI-{ts_val}")

        # Filter to available columns and drop NaN on all needed vars
        groups_avail = {}
        all_cols = []
        for gname, gcols in GROUP_DEFS.items():
            avail = [c for c in gcols if c in edf.columns]
            groups_avail[gname] = avail
            all_cols += avail
            ts_print(f"    {gname}: {len(avail)}/{len(gcols)} vars")

        work = edf[['GAGE_ID'] + all_cols + [TARGET_USE]].dropna()
        ts_print(f"  Clean events: {len(work)} ({100*len(work)/len(edf):.1f}%)")

        if len(work) < 50:
            ts_print("  SKIP – too few clean events")
            continue

        X_raw = work[all_cols]
        y = work[TARGET_USE].values
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X_raw), columns=all_cols, index=work.index)

        # B1 – Sequential hierarchical OLS
        ts_print("  Sequential hierarchical OLS:")
        cum_cols = []
        r2s, dr2s = [], []
        for gname in group_order:
            cum_cols += groups_avail[gname]
            mdl = LinearRegression().fit(X[cum_cols], y)
            r2 = mdl.score(X[cum_cols], y)
            dr2 = r2 - (r2s[-1] if r2s else 0)
            r2s.append(r2)
            dr2s.append(dr2)
            ts_print(f"      +{gname:25s} → R²={r2:.4f}  ΔR²={dr2:.4f}")

        # B2 – Shapley-value variance partitioning (5! = 120 permutations)
        ts_print("  Shapley partitioning (120 permutations) …")
        shapley = {gn: [] for gn in group_order}
        n_perms = 0
        for perm in permutations(group_order):
            r2_prev = 0
            for i, gn in enumerate(perm):
                cols_so_far = []
                for g in perm[:i+1]:
                    cols_so_far += groups_avail[g]
                if not cols_so_far:
                    continue
                mdl = LinearRegression().fit(X[cols_so_far], y)
                r2_now = mdl.score(X[cols_so_far], y)
                shapley[gn].append(r2_now - r2_prev)
                r2_prev = r2_now
            n_perms += 1

        ts_print(f"    Completed {n_perms} permutations")

        shap_mean = {gn: np.nanmean(v) if v else 0 for gn, v in shapley.items()}
        shap_std = {gn: np.nanstd(v) if v else 0 for gn, v in shapley.items()}
        shap_min = {gn: np.nanmin(v) if v else 0 for gn, v in shapley.items()}
        shap_max = {gn: np.nanmax(v) if v else 0 for gn, v in shapley.items()}

        for gn in group_order:
            ts_print(f"      {gn:25s}: Shapley ΔR²={shap_mean[gn]:.4f} ± {shap_std[gn]:.4f}")

        hier_results[ts_val] = dict(
            r2s=r2s, dr2s=dr2s,
            shap_mean=shap_mean, shap_std=shap_std, shap_min=shap_min, shap_max=shap_max,
            shapley=shapley, X=X, y=y, work=work,
            groups_avail=groups_avail
        )

        # Save Shapley values
        for gn in group_order:
            for perm_idx, val in enumerate(shapley[gn]):
                shapley_df_list.append({
                    'timescale': ts_val,
                    'group': gn,
                    'permutation_idx': perm_idx,
                    'marginal_r2': val
                })

        # Report
        report_lines.append(f"\nTimescale SPEI-{ts_val} / SSI-{ts_val} (event-level, n={len(work):,}):")
        for i, gn in enumerate(group_order):
            report_lines.append(f"  +{gn}: R²={r2s[i]:.4f} ΔR²={dr2s[i]:.4f} Shapley={shap_mean[gn]:.4f}")

    if shapley_df_list:
        shapley_df = pd.DataFrame(shapley_df_list)
        shapley_df.to_csv(OUT / 'shapley_values.csv', index=False)
        ts_print(f"Saved Shapley values: {len(shapley_df)} entries")

    # ══════════════════════════════════════════════════════════════════
    # PART C – RANDOM FOREST (EVENT LEVEL)
    # ══════════════════════════════════════════════════════════════════
    ts_print("\n═══ PART C: RANDOM FOREST (EVENT LEVEL) ═══")

    rf_results = {}
    for ts_val in [3, 6]:
        if ts_val not in hier_results:
            continue
        hr = hier_results[ts_val]
        X, y = hr['X'], hr['y']
        ts_print(f"\n  Timescale {ts_val}: {X.shape[0]:,} events × {X.shape[1]} features")

        rf = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=20,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X, y)
        cv = cross_val_score(rf, X, y, cv=5, scoring='r2')
        gc.collect()
        ts_print(f"  5-fold CV R²: {cv.mean():.4f} ± {cv.std():.4f}")
        ts_print(f"  Train R²: {rf.score(X, y):.4f}")

        fi = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)

        # Grouped importance
        gi = {}
        for gn, gcols in hr['groups_avail'].items():
            gi[gn] = fi.loc[fi['feature'].isin(gcols), 'importance'].sum()

        ts_print(f"  Grouped MDI: {gi}")

        rf_results[ts_val] = dict(rf=rf, cv=cv, fi=fi, gi=gi, train_r2=rf.score(X, y))

        report_lines.append(f"\nRF Timescale {ts_val} (event-level):")
        report_lines.append(f"  CV R² = {cv.mean():.4f} ± {cv.std():.4f}")
        report_lines.append(f"  Train R² = {rf.score(X, y):.4f}")
        for gn in group_order:
            report_lines.append(f"    {gn} MDI={gi.get(gn, 0):.4f}")

    # ══════════════════════════════════════════════════════════════════
    # PART D – RECOVERY REGIME CLASSIFICATION (BASIN LEVEL)
    # ══════════════════════════════════════════════════════════════════
    ts_print("\n═══ PART D: RECOVERY REGIME CLASSIFICATION (BASIN LEVEL) ═══")

    p6 = pairs[pairs['timescale'] == 6].copy()

    def _basin_regime(grp):
        n = len(grp)
        # Use dynamic BFI (median across events) for regime classification
        bfi_col = 'dyn_BFI' if 'dyn_BFI' in grp.columns else 'BFI'
        rk_col = 'dyn_recession_k_days' if 'dyn_recession_k_days' in grp.columns else 'recession_k_days'
        bfi = grp[bfi_col].median() if bfi_col in grp.columns else np.nan
        rk = grp[rk_col].median() if rk_col in grp.columns else np.nan

        # Correlation between rebound and recovery lag
        valid = grp[['spei_rebound_6mo', TARGET_USE]].dropna()
        if len(valid) > 4:
            corr, _ = spearmanr(valid['spei_rebound_6mo'], valid[TARGET_USE])
        else:
            corr = np.nan

        # Classification logic
        if pd.notna(corr) and corr < -0.3 and pd.notna(bfi) and bfi < 0.4:
            regime = 'Climate-controlled'
        elif pd.notna(bfi) and bfi > 0.6 and pd.notna(rk) and rk > 20:
            regime = 'Memory-controlled'
        else:
            regime = 'Mixed'

        return pd.Series({
            'n_events': n,
            'corr_rebound': corr,
            'BFI': bfi,
            'recession_k': rk,
            'regime': regime
        })

    regime_df = p6.groupby('GAGE_ID').apply(_basin_regime).reset_index()
    regime_df = regime_df[regime_df['n_events'] >= 3]
    regime_df.to_csv(OUT / 'basin_recovery_regimes.csv', index=False)

    ts_print(f"  {len(regime_df)} basins with ≥3 events:")
    for r in ['Climate-controlled', 'Memory-controlled', 'Mixed']:
        n = (regime_df['regime'] == r).sum()
        pct = 100 * n / len(regime_df) if len(regime_df) > 0 else 0
        ts_print(f"    {r}: {n} ({pct:.1f}%)")

    # Merge back to pairs
    pairs = pairs.merge(regime_df[['GAGE_ID', 'regime']], on='GAGE_ID', how='left')

    # ══════════════════════════════════════════════════════════════════
    # PART E – OPERATIONAL PREDICTOR (RIDGE + RF)
    # ══════════════════════════════════════════════════════════════════
    ts_print("\n═══ PART E: OPERATIONAL PREDICTOR (RIDGE + RF) ═══")

    # Candidate variables (from all 5 groups)
    cand_vars = []
    for gcols in GROUP_DEFS.values():
        cand_vars.extend(gcols)
    cand_vars = list(set(cand_vars))
    cand_vars = [c for c in cand_vars if c in pairs.columns]

    # Also add interaction terms
    int_terms = ['ARIDITY_INDEX_x_BFI', 'ARIDITY_INDEX_x_recession_k']

    # Filter to SPEI-6 events with complete data
    op_base = pairs[pairs['timescale'] == 6][cand_vars + [TARGET_USE]].dropna()
    ts_print(f"  {len(op_base):,} clean SPEI-6 events")

    if len(op_base) > 50000:
        op_base = op_base.sample(50000, random_state=42)
        ts_print(f"  Sampled to {len(op_base):,}")

    X_op = op_base[cand_vars].copy()
    y_op = op_base[TARGET_USE].values

    # Add interactions
    if 'ARIDITY_INDEX' in X_op.columns and 'dyn_BFI' in X_op.columns:
        X_op['ARIDITY_INDEX_x_BFI'] = X_op['ARIDITY_INDEX'] * X_op['dyn_BFI']
    if 'ARIDITY_INDEX' in X_op.columns and 'dyn_recession_k_days' in X_op.columns:
        X_op['ARIDITY_INDEX_x_recession_k'] = X_op['ARIDITY_INDEX'] * X_op['dyn_recession_k_days']

    # Standardize
    sc_op = StandardScaler()
    X_op_s = pd.DataFrame(
        sc_op.fit_transform(X_op),
        columns=X_op.columns,
        index=X_op.index
    )

    # Ridge regression with CV
    ts_print("  Ridge regression (RidgeCV) …")
    ridge = RidgeCV(alphas=[0.01, 0.1, 1, 10, 100], cv=5)
    ridge.fit(X_op_s, y_op)
    ridge_r2 = ridge.score(X_op_s, y_op)
    ridge_cv = cross_val_score(ridge, X_op_s, y_op, cv=5, scoring='r2')
    ts_print(f"    Train R²: {ridge_r2:.4f}")
    ts_print(f"    CV R²: {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")
    ts_print(f"    Selected α: {ridge.alpha_:.4f}")

    # Random Forest for operational model
    ts_print("  Random Forest (operational) …")
    rf_op = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    rf_op.fit(X_op_s, y_op)
    rf_op_r2 = rf_op.score(X_op_s, y_op)
    rf_op_cv = cross_val_score(rf_op, X_op_s, y_op, cv=5, scoring='r2')
    ts_print(f"    Train R²: {rf_op_r2:.4f}")
    ts_print(f"    CV R²: {rf_op_cv.mean():.4f} ± {rf_op_cv.std():.4f}")

    # Store results
    op_results = dict(
        ridge_model=ridge,
        rf_model=rf_op,
        ridge_r2=ridge_r2,
        ridge_cv=ridge_cv,
        rf_op_r2=rf_op_r2,
        rf_op_cv=rf_op_cv,
        X_op=X_op_s,
        y_op=y_op,
        features=X_op.columns.tolist(),
        scaler=sc_op
    )

    # Ridge coefficients
    ridge_coeff = pd.DataFrame({
        'variable': X_op.columns,
        'beta': ridge.coef_,
        'abs_beta': np.abs(ridge.coef_)
    }).sort_values('abs_beta', ascending=False)
    ridge_coeff.to_csv(OUT / 'operational_ridge_coeff.csv', index=False)

    # RF importance
    rf_op_fi = pd.DataFrame({
        'variable': X_op.columns,
        'importance': rf_op.feature_importances_
    }).sort_values('importance', ascending=False)
    rf_op_fi.to_csv(OUT / 'operational_rf_importance.csv', index=False)

    report_lines.append(f"\nOperational Predictor:")
    report_lines.append(f"  Ridge Regression: CV R² = {ridge_cv.mean():.4f} ± {ridge_cv.std():.4f}")
    report_lines.append(f"  Random Forest:    CV R² = {rf_op_cv.mean():.4f} ± {rf_op_cv.std():.4f}")
    report_lines.append(f"  Features: {len(X_op.columns)}")

    # ══════════════════════════════════════════════════════════════════
    # PART F – PUBLICATION FIGURES
    # ══════════════════════════════════════════════════════════════════
    gc.collect()
    ts_print("\n═══ PART F: PUBLICATION FIGURES ═══")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import Patch
    from scipy.ndimage import uniform_filter1d
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.6,
    })

    ALBERS = ccrs.AlbersEqualArea(
        central_longitude=-96,
        central_latitude=37.5,
        standard_parallels=(29.5, 45.5)
    )
    PC = ccrs.PlateCarree()

    # Pre-load US geometries
    _US_GEOM = _CANADA_GEOM = _MEXICO_GEOM = None
    try:
        import geopandas as _gpd
        _world = _gpd.read_file(_gpd.datasets.get_path('naturalearth_lowres'))
        _us = _world[_world['name'] == 'United States of America'].geometry.values
        _ca = _world[_world['name'] == 'Canada'].geometry.values
        _mx = _world[_world['name'] == 'Mexico'].geometry.values
        _US_GEOM = _us[0] if len(_us) > 0 else None
        _CANADA_GEOM = _ca[0] if len(_ca) > 0 else None
        _MEXICO_GEOM = _mx[0] if len(_mx) > 0 else None
    except Exception:
        pass

    def _make_conus_ax(fig, pos):
        """Create CONUS map axis with country outlines."""
        if isinstance(pos, tuple):
            ax = fig.add_subplot(*pos, projection=ALBERS)
        else:
            ax = fig.add_subplot(pos, projection=ALBERS)
        ax.set_extent([-125, -66.5, 24.5, 49.5], crs=PC)
        ax.set_facecolor('#e8f0f8')

        if _CANADA_GEOM is not None:
            ax.add_geometries([_CANADA_GEOM], crs=PC, facecolor='#ededed',
                              edgecolor='#888888', linewidth=0.5, zorder=1)
        if _MEXICO_GEOM is not None:
            ax.add_geometries([_MEXICO_GEOM], crs=PC, facecolor='#ededed',
                              edgecolor='#888888', linewidth=0.5, zorder=1)
        if _US_GEOM is not None:
            ax.add_geometries([_US_GEOM], crs=PC, facecolor='#f9f9f9',
                              edgecolor='#333333', linewidth=0.7, zorder=2)

        gl = ax.gridlines(draw_labels=False, linewidth=0.3, color='#cccccc',
                           alpha=0.5, linestyle='--')
        return ax

    def _save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(OUT / f'{name}.{ext}', dpi=300, facecolor='white', bbox_inches='tight')
        plt.close(fig)
        ts_print(f"  Saved {name}")

    def _conus_filter(df):
        v = df.dropna(subset=['LNG_GAGE', 'LAT_GAGE']).copy()
        return v[
            (v['LNG_GAGE'] > -130) & (v['LNG_GAGE'] < -65) &
            (v['LAT_GAGE'] > 24) & (v['LAT_GAGE'] < 50)
        ]

    # ── FIG 9: Hierarchical R² + Shapley (5-group, 2-panel) ────
    ts_print("Fig 9 …")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ti, ts_val in enumerate([3, 6]):
        if ts_val not in hier_results:
            continue
        ax = axes[ti]
        hr = hier_results[ts_val]

        # Sequential ΔR²
        x_left = np.arange(len(group_order)) - 0.2
        dr2s_vals = hr['dr2s']
        colors_seq = [GROUP_COLORS[gn] for gn in group_order]
        ax.bar(x_left, dr2s_vals, width=0.35, label='Sequential ΔR²',
               color=colors_seq, edgecolor='white', linewidth=0.8, alpha=0.8)

        # Shapley values
        x_right = np.arange(len(group_order)) + 0.2
        shap_vals = [hr['shap_mean'][gn] for gn in group_order]
        ax.bar(x_right, shap_vals, width=0.35, label='Shapley mean ΔR²',
               color=colors_seq, edgecolor='#333333', linewidth=0.8, hatch='//', alpha=0.7)

        # Annotate
        for i, (x, val) in enumerate(zip(x_left, dr2s_vals)):
            ax.text(x, val + 0.005, f'{val:.3f}', ha='center', va='bottom',
                   fontsize=7, fontweight='bold')
        for i, (x, val) in enumerate(zip(x_right, shap_vals)):
            ax.text(x, val + 0.005, f'{val:.3f}', ha='center', va='bottom',
                   fontsize=7, fontweight='bold')

        ax.set_xticks(np.arange(len(group_order)))
        ax.set_xticklabels([gn.replace('_', ' ') for gn in group_order],
                            fontsize=8, rotation=15, ha='right')
        ax.set_ylabel('ΔR²', fontsize=10, fontweight='bold')
        ax.set_title(f'SPEI-{ts_val} / SSI-{ts_val}', fontsize=11, fontweight='bold')
        ax.set_ylim(0, max(dr2s_vals + shap_vals) * 1.3)
        ax.legend(loc='upper right', fontsize=8, frameon=True)
        ax.grid(axis='y', alpha=0.15)

    fig.suptitle('Event-level hierarchical variance decomposition & Shapley-value partitioning',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    _save(fig, 'fig9_hierarchical_r2_shapley')

    # ── FIG 10: Shapley Group Decomposition (OLS vs RF, 5 groups) ────
    ts_print("Fig 10 …")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for ti, ts_val in enumerate([3, 6]):
        if ts_val not in hier_results:
            continue
        ax = axes[ti]
        hr = hier_results[ts_val]
        rf = rf_results.get(ts_val)

        shap_means = [hr['shap_mean'][gn] for gn in group_order]
        order = np.argsort(shap_means)[::-1]
        names_sorted = [group_order[i] for i in order]
        shap_sorted = [shap_means[i] for i in order]
        colors_sorted = [GROUP_COLORS[n] for n in names_sorted]

        x_pos = np.arange(len(names_sorted))

        # OLS
        ax.barh(x_pos - 0.2, shap_sorted, height=0.35, label='OLS Shapley',
               color=colors_sorted, edgecolor='#333', linewidth=0.6, alpha=0.85)

        # RF
        if rf:
            gi = rf['gi']
            rf_sorted = [gi.get(group_order[i], 0) for i in order]
            ax.barh(x_pos + 0.2, rf_sorted, height=0.35, label='RF grouped MDI',
                   color=colors_sorted, edgecolor='#333', linewidth=0.6, hatch='xx', alpha=0.7)

        ax.set_yticks(x_pos)
        ax.set_yticklabels([n.replace('_', ' ') for n in names_sorted], fontsize=9)
        ax.set_xlabel('Mean ΔR² (OLS) or MDI (RF)', fontsize=9)
        ax.set_title(f'SPEI-{ts_val} / SSI-{ts_val}', fontsize=10, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, frameon=True)
        ax.grid(axis='x', alpha=0.15)

    fig.suptitle('Comparison of OLS Shapley and Random Forest group importance (5 groups)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    _save(fig, 'fig10_shapley_group_decomposition')

    # ── FIG 11: Recovery Regime Map (2×2 CONUS panels) ────
    ts_print("Fig 11 …")
    # Get coords from pairs (check column existence first)
    coord_cols = [c for c in ['GAGE_ID', 'LAT_GAGE', 'LNG_GAGE'] if c in pairs.columns]
    if len(coord_cols) == 3:
        coords_for_map = pairs[coord_cols].drop_duplicates('GAGE_ID')
    else:
        # Fallback: load coords from GAGES-II directly
        ts_print(f"  WARNING: coord columns missing from pairs ({pairs.columns.tolist()[:5]}...)")
        coords_for_map = _read_gages_csv('conterm_basinid.txt', ['LAT_GAGE', 'LNG_GAGE'])
    rc = regime_df.merge(coords_for_map, on='GAGE_ID', how='left')
    rc = _conus_filter(rc)

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.15, wspace=0.05)

    REG_COLS = {
        'Climate-controlled': '#2166ac',
        'Memory-controlled': '#b2182b',
        'Mixed': '#aaaaaa'
    }

    # (a) Regime map
    ax = _make_conus_ax(fig, gs[0, 0])
    for reg, col in REG_COLS.items():
        m = rc['regime'] == reg
        s = rc[m]
        ax.scatter(s['LNG_GAGE'].values, s['LAT_GAGE'].values,
                  c=col, s=5, alpha=0.75, label=f'{reg} (n={m.sum()})',
                  edgecolors='none', rasterized=True, transform=PC, zorder=4)
    ax.set_title('(a) Recovery regime classification', fontsize=11, fontweight='bold', pad=5)
    ax.legend(loc='lower left', fontsize=7.5, frameon=True, framealpha=0.92,
             markerscale=2.5, handletextpad=0.4, edgecolor='grey')

    # (b) BFI map
    ax = _make_conus_ax(fig, gs[0, 1])
    v = rc.dropna(subset=['BFI']).sort_values('BFI')
    sc2 = ax.scatter(v['LNG_GAGE'].values, v['LAT_GAGE'].values,
                    c=v['BFI'].values, cmap='RdYlBu', s=5, alpha=0.8,
                    vmin=0, vmax=1, edgecolors='none', rasterized=True,
                    transform=PC, zorder=4)
    cb = plt.colorbar(sc2, ax=ax, shrink=0.65, pad=0.02, aspect=18)
    cb.set_label('BFI', fontsize=9)
    ax.set_title('(b) Baseflow Index (catchment memory)', fontsize=11, fontweight='bold', pad=5)

    # (c) Recovery lag map (event-level median per basin)
    ax = _make_conus_ax(fig, gs[1, 0])
    p6_map = pairs[pairs['timescale'] == 6].copy()
    if 'LAT_GAGE' in p6_map.columns and 'LNG_GAGE' in p6_map.columns:
        p6_basin = p6_map.groupby('GAGE_ID').agg(
            rec_lag_med=(TARGET_USE, 'median'),
            LAT_GAGE=('LAT_GAGE', 'first'),
            LNG_GAGE=('LNG_GAGE', 'first')
        ).reset_index()
    else:
        p6_basin = p6_map.groupby('GAGE_ID').agg(
            rec_lag_med=(TARGET_USE, 'median'),
        ).reset_index()
        p6_basin = p6_basin.merge(coords_for_map, on='GAGE_ID', how='left')
    p6_basin = _conus_filter(p6_basin)
    v3 = p6_basin.dropna(subset=['rec_lag_med']).sort_values('rec_lag_med')
    sc3 = ax.scatter(v3['LNG_GAGE'].values, v3['LAT_GAGE'].values,
                    c=v3['rec_lag_med'].clip(0, 15).values,
                    cmap='RdYlBu_r', s=5, alpha=0.8, vmin=0, vmax=15,
                    edgecolors='none', rasterized=True, transform=PC, zorder=4)
    cb3 = plt.colorbar(sc3, ax=ax, shrink=0.65, pad=0.02, aspect=18)
    cb3.set_label('Months', fontsize=9)
    ax.set_title('(c) Median recovery lag (months)', fontsize=11, fontweight='bold', pad=5)

    # (d) Recession k map
    ax = _make_conus_ax(fig, gs[1, 1])
    v4 = rc.dropna(subset=['recession_k']).sort_values('recession_k')
    sc4 = ax.scatter(v4['LNG_GAGE'].values, v4['LAT_GAGE'].values,
                    c=v4['recession_k'].clip(0, 60).values,
                    cmap='YlOrRd', s=5, alpha=0.8, vmin=0, vmax=60,
                    edgecolors='none', rasterized=True, transform=PC, zorder=4)
    cb4 = plt.colorbar(sc4, ax=ax, shrink=0.65, pad=0.02, aspect=18)
    cb4.set_label('Days', fontsize=9)
    ax.set_title('(d) Recession constant (days)', fontsize=11, fontweight='bold', pad=5)

    fig.suptitle('Recovery regime classification and underlying catchment properties',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.text(0.5, 0.005, f'{len(rc):,} basins · SPEI-6/SSI-6',
             ha='center', fontsize=8, style='italic', color='#555')
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    _save(fig, 'fig11_recovery_regime_map')

    # ── FIG 12: Memory Signature (hexbin panels) ────
    ts_print("Fig 12 …")
    p6_mem = pairs[pairs['timescale'] == 6].copy()

    panels = [
        ('dyn_BFI', TARGET_USE, 'Dynamic BFI', '#b2182b', 'Reds'),
        ('dyn_recession_k_days', TARGET_USE, 'Dynamic Recession k (days)', '#4393c3', 'Blues'),
        ('cv_Q', TARGET_USE, 'CV of discharge', '#1a9850', 'Greens'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    letters = 'abc'

    for pi, (xvar, yvar, xlabel_label, col, cmap) in enumerate(panels):
        ax = axes[pi]
        if xvar not in p6_mem.columns or yvar not in p6_mem.columns:
            ax.set_visible(False)
            continue

        d = p6_mem.dropna(subset=[xvar, yvar])
        if len(d) == 0:
            ax.set_visible(False)
            continue

        hb = ax.hexbin(d[xvar], d[yvar].clip(-2, 30), gridsize=35,
                       cmap=cmap, alpha=0.8, mincnt=1,
                       linewidths=0.1, edgecolors='white')

        # Smoothed trend
        try:
            s = d.sort_values(xvar)
            w = max(len(s) // 15, 15)
            sm = uniform_filter1d(s[yvar].values.astype(float), w)
            ax.plot(s[xvar].values, sm, 'k-', linewidth=2.5, zorder=5)
        except Exception:
            pass

        rho, pv = spearmanr(d[xvar], d[yvar])
        sig = '***' if pv < 0.001 else '**' if pv < 0.01 else '*' if pv < 0.05 else ''

        ax.set_xlabel(xlabel_label, fontsize=10, fontweight='bold')
        ax.set_ylabel('Recovery lag (months)', fontsize=10, fontweight='bold')
        ax.set_title(f'({letters[pi]}) {xlabel_label}: ρ = {rho:+.3f}{sig} (n={len(d):,})',
                    fontweight='bold', fontsize=10)
        ax.set_ylim(-1, min(d[yvar].quantile(0.98) * 1.3, 25) if len(d) > 0 else 20)
        ax.grid(alpha=0.1)
        plt.colorbar(hb, ax=ax, label='count')

    fig.suptitle('Catchment memory signatures and event-level drought recovery (SPEI-6)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    _save(fig, 'fig12_memory_signature')

    # ── FIG 13: Operational Predictor Performance (Ridge + RF) ────
    ts_print("Fig 13 …")
    fig = plt.figure(figsize=(16, 5))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)

    # (a) Ridge: Obs vs Pred
    ax = fig.add_subplot(gs[0])
    ridge_pred = op_results['ridge_model'].predict(op_results['X_op'])
    ax.hexbin(op_results['y_op'], ridge_pred, gridsize=50,
             cmap='YlOrRd', mincnt=1, linewidths=0.1, edgecolors='white')
    lims = [0, max(np.percentile(op_results['y_op'], 99),
                  np.percentile(ridge_pred, 99))]
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line', zorder=5)
    ax.set_xlabel('Observed recovery lag (months)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted recovery lag (months)', fontsize=10, fontweight='bold')
    ax.set_title(f'(a) Ridge regression\nCV R² = {op_results["ridge_cv"].mean():.3f}',
                fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # (b) RF: Obs vs Pred
    ax = fig.add_subplot(gs[1])
    rf_pred = op_results['rf_model'].predict(op_results['X_op'])
    ax.hexbin(op_results['y_op'], rf_pred, gridsize=50,
             cmap='YlOrRd', mincnt=1, linewidths=0.1, edgecolors='white')
    ax.plot(lims, lims, 'k--', linewidth=2, label='1:1 line', zorder=5)
    ax.set_xlabel('Observed recovery lag (months)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Predicted recovery lag (months)', fontsize=10, fontweight='bold')
    ax.set_title(f'(b) Random Forest\nCV R² = {op_results["rf_op_cv"].mean():.3f}',
                fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # (c) Coefficients (Ridge)
    ax = fig.add_subplot(gs[2])
    cd = ridge_coeff.head(10).sort_values('beta')
    colors_c = ['#d73027' if b < 0 else '#4575b4' for b in cd['beta']]
    PRETTY = {
        'ARIDITY_INDEX': 'Aridity', 'PET': 'PET',
        'propagation_lag_months': 'Propagation lag',
        'spei_severity': 'SPEI severity', 'spei_duration': 'SPEI duration',
        'dyn_recession_k_days': 'Dyn Recession k', 'dyn_cv_Q': 'Dyn CV discharge',
        'dyn_BFI': 'Dyn Baseflow Index', 'dyn_zero_flow_fraction': 'Dyn Zero flow',
        'recession_k_days': 'Recession k', 'cv_Q': 'CV discharge',
        'BFI': 'Baseflow Index', 'PPTAVG_BASIN': 'Precipitation',
        'FORESTNLCD06': 'Forest cover', 'PERMAVE': 'Permeability',
        'ELEV_MEAN_M_BASIN': 'Elevation', 'SNOW_PCT_PRECIP': 'Snow fraction',
        'spei_rebound_6mo': 'SPEI rebound (6mo)',
        'ARIDITY_INDEX_x_BFI': 'Aridity × BFI',
        'ARIDITY_INDEX_x_recession_k': 'Aridity × Recession k',
    }
    labels_c = [PRETTY.get(v, v) for v in cd['variable']]
    ax.barh(range(len(cd)), cd['beta'], color=colors_c, edgecolor='#333',
           linewidth=0.5, height=0.6)
    ax.set_yticks(range(len(cd)))
    ax.set_yticklabels(labels_c, fontsize=9)
    ax.set_xlabel('Standardized coefficient β', fontsize=10, fontweight='bold')
    ax.set_title('(c) Ridge coefficients (top 10)', fontweight='bold', fontsize=11)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.grid(axis='x', alpha=0.15)

    for i, (_, row) in enumerate(cd.iterrows()):
        ax.text(row['beta'] + (0.01 if row['beta'] >= 0 else -0.01), i,
               f'{row["beta"]:+.3f}', va='center', fontsize=7,
               ha='left' if row['beta'] >= 0 else 'right', fontweight='bold')

    fig.suptitle('Operational predictors: Ridge regression and Random Forest (event-level)',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    _save(fig, 'fig13_operational_predictor')

    # ══════════════════════════════════════════════════════════════════
    # SUMMARY REPORT
    # ══════════════════════════════════════════════════════════════════
    ts_print("\n═══ GENERATING SUMMARY REPORT ═══")

    report_lines.insert(0, "STEP 4: EVENT-LEVEL HIERARCHICAL ATTRIBUTION – SUMMARY")
    report_lines.insert(1, "=" * 80)

    report_lines.append(f"\nDATA SUMMARY:")
    report_lines.append(f"  Event-level analysis (not basin medians)")
    report_lines.append(f"  Total events: {len(pairs):,}")
    report_lines.append(f"  Basins: {pairs['GAGE_ID'].nunique()}")
    report_lines.append(f"  SPEI-3 events: {len(pairs[pairs['timescale']==3]):,}")
    report_lines.append(f"  SPEI-6 events: {len(pairs[pairs['timescale']==6]):,}")

    report_lines.append(f"\nRECOVERY REGIMES (n={len(regime_df)} basins with ≥3 events):")
    for r in ['Climate-controlled', 'Memory-controlled', 'Mixed']:
        n = (regime_df['regime'] == r).sum()
        pct = 100 * n / len(regime_df) if len(regime_df) > 0 else 0
        report_lines.append(f"  {r}: {n} ({pct:.1f}%)")

    report_lines.append(f"\nOPERATIONAL MODELS (event-level):")
    report_lines.append(f"  Ridge Regression:")
    report_lines.append(f"    CV R²: {op_results['ridge_cv'].mean():.4f} ± {op_results['ridge_cv'].std():.4f}")
    report_lines.append(f"    Selected α: {op_results['ridge_model'].alpha_:.4f}")
    report_lines.append(f"  Random Forest:")
    report_lines.append(f"    CV R²: {op_results['rf_op_cv'].mean():.4f} ± {op_results['rf_op_cv'].std():.4f}")

    report_lines.append(f"\nGROUP DEFINITIONS (5 groups, 120 Shapley permutations):")
    for gname, gcols in GROUP_DEFS.items():
        report_lines.append(f"  {gname}: {len(gcols)} variables")
        report_lines.append(f"    {', '.join(gcols[:3])}{'...' if len(gcols) > 3 else ''}")

    report_lines.append(f"\nOUTPUTS SAVED TO: {OUT}/")
    report_lines.append(f"  - assembled_event_data.csv (all {len(pairs):,} events)")
    report_lines.append(f"  - basin_recovery_regimes.csv")
    report_lines.append(f"  - shapley_values.csv (120 permutations × 5 groups)")
    report_lines.append(f"  - operational_ridge_coeff.csv")
    report_lines.append(f"  - operational_rf_importance.csv")
    report_lines.append(f"  - 5 figures: fig9–fig13 (PNG + PDF)")

    (OUT / 'analysis_summary.txt').write_text("\n".join(report_lines))
    ts_print(f"\nAll outputs saved to: {OUT}/")
    ts_print("DONE")

if __name__ == '__main__':
    main()
