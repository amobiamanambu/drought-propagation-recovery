#!/usr/bin/env python3
"""
XGBoost + SHAP Analysis for Drought Propagation/Recovery
=========================================================
Produces all outputs needed to replace stepwise regression results in:
  - Figure 4: SHAP importance bars + XGBoost R² for scatter panels
  - Figure 5: Top predictor rankings with SHAP direction
  - Figure 6: Ecoregion-specific R² maps
  - Figure 7: Storage duality (dyn_BFI SHAP) + Ref vs Non-ref R²

Exports CSV files that can be directly used by existing plotting scripts.

Hyperparameters documented at end of output.
"""
import warnings, os, time, json
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb
import shap

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'XGBoost_Results')
os.makedirs(OUT, exist_ok=True)

t_global = time.time()

# ═══════════════════════════════════════════════════════════════
#  HYPERPARAMETERS (fixed across all models)
# ═══════════════════════════════════════════════════════════════
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=20,
    reg_alpha=0.1,        # L1 regularization
    reg_lambda=1.0,       # L2 regularization
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

CV_FOLDS = 5

print("=" * 70)
print("  XGBoost + SHAP ANALYSIS")
print("  Hyperparameters:")
for k, v in XGB_PARAMS.items():
    if k not in ('n_jobs', 'verbosity', 'random_state'):
        print(f"    {k}: {v}")
print(f"  CV folds: {CV_FOLDS}")
print("=" * 70)


# ═══════════════════════════════════════════════════════════════
#  1. LOAD DATA
# ═══════════════════════════════════════════════════════════════
print("\n[1] Loading data...")

events = pd.read_csv(os.path.join(BASE, 'Data', 'matched_pairs_dynamic.csv'),
                      dtype={'GAGE_ID': str})
events['GAGE_ID'] = events['GAGE_ID'].str.strip().str.zfill(8)

basins = pd.read_csv(os.path.join(BASE, 'Data', 'drought_basin_classification.csv'),
                      dtype={'GAGE_ID': str})
basins['GAGE_ID'] = basins['GAGE_ID'].str.strip().str.zfill(8)

# GAGES-II static attributes
gages_dir = os.path.join(BASE, 'gagesii_csv')
static_files = ['conterm_hydro.txt', 'conterm_climate.txt', 'conterm_topo.txt',
                'conterm_bas_classif.txt', 'conterm_bas_morph.txt',
                'conterm_lc06_basin.txt', 'conterm_hydromod_dams.txt']

STATIC_COLS = [
    'STREAMS_KM_SQ_KM','STRAHLER_MAX','CONTACT','PERDUN','PERHOR','TOPWET','RUNAVE7100',
    'PPTAVG_BASIN','T_AVG_BASIN','RH_BASIN','PET','SNOW_PCT_PRECIP','PRECIP_SEAS_IND',
    'ELEV_MEAN_M_BASIN','DRAIN_SQKM','SLOPE_PCT','RRMEAN','ASPECT_EASTNESS','ASPECT_NORTHNESS',
    'GEOL_REEDBUSH_DOM_PCT','HGA','HGB','HGC','HGD','PERMAVE','RFACT','BDAVE','AWCAVE',
    'ROCKDEPAVE','WTDEPAVE','BAS_COMPACTNESS',
    'FORESTNLCD06','CROPSNLCD06','GRASSNLCD06','SHRUBNLCD06','WOODYWETNLCD06','EMERGWETNLCD06','DEVNLCD06',
    'NDAMS_2009','STOR_NID_2009','RAW_DIS_NEAREST_DAM','RAW_AVG_DIS_ALLDAMS','RAW_DIS_NEAREST_MAJ_DAM',
]
HUMAN_VARS = {'NDAMS_2009','STOR_NID_2009','RAW_DIS_NEAREST_DAM',
              'RAW_AVG_DIS_ALLDAMS','RAW_DIS_NEAREST_MAJ_DAM','DEVNLCD06'}

static_df = None
for fname in static_files:
    fpath = os.path.join(gages_dir, fname)
    if not os.path.exists(fpath): continue
    df = pd.read_csv(fpath, dtype={'STAID': str}, encoding='latin-1')
    if 'STAID' not in df.columns: continue
    df['STAID'] = df['STAID'].str.strip().str.zfill(8)
    cols = [c for c in STATIC_COLS if c in df.columns]
    if cols:
        sub = df[['STAID'] + cols].copy()
        static_df = sub if static_df is None else static_df.merge(sub, on='STAID', how='outer')
static_df = static_df.rename(columns={'STAID': 'GAGE_ID'})

# Merge static into events (preserve CLASS and ecoregion from events)
events_m = events.merge(static_df, on='GAGE_ID', how='left')

print(f"  Events: {len(events_m):,} | Basins: {len(basins):,} | Static vars: {len(static_df.columns)-1}")


# ═══════════════════════════════════════════════════════════════
#  2. SCREENING FUNCTION
# ═══════════════════════════════════════════════════════════════
def screen_preds(X_df):
    """Screen: >20% missing, zero-var, |r|>=0.70, VIF>10"""
    cols = list(X_df.columns)
    miss = X_df.isnull().mean()
    std = X_df.std()
    cols = [c for c in cols if miss[c] < 0.20 and std[c] > 1e-6]
    # Correlation
    n = len(X_df)
    Xsub = X_df[cols].sample(min(n, 5000), random_state=42) if n > 5000 else X_df[cols]
    R = Xsub.corr().abs().values
    drop = set()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            if R[i,j] >= 0.70 and cols[j] not in drop:
                drop.add(cols[j])
    cols = [c for c in cols if c not in drop]
    # VIF
    for _ in range(5):
        if len(cols) < 3: break
        Xcc = X_df[cols].dropna()
        if len(Xcc) > 5000: Xcc = Xcc.sample(5000, random_state=42)
        try:
            Xn = (Xcc - Xcc.mean()) / Xcc.std()
            C = Xn.T @ Xn / (len(Xn)-1) + np.eye(len(cols))*1e-8
            vif = np.diag(np.linalg.inv(C))
            worst = np.argmax(vif)
            if vif[worst] > 10: cols.pop(worst)
            else: break
        except: break
    return cols


# ═══════════════════════════════════════════════════════════════
#  3. HELPER: fit XGBoost, compute SHAP, return results
# ═══════════════════════════════════════════════════════════════
DYN_COLS = ['dyn_BFI', 'dyn_recession_k_days', 'dyn_cv_Q', 'antecedent_ssi']
EVENT_SEVERITY = ['spei_intensity', 'ssi_intensity', 'spei_false_rec', 'ssi_false_rec']

def classify_predictor(name):
    """Classify predictor as DYN, EVENT, or STATIC (matching MATLAB Type column)."""
    if name in DYN_COLS or name.startswith('mean_dyn_') or name == 'mean_antecedent_ssi':
        return 'DYN'
    elif name in EVENT_SEVERITY:
        return 'EVENT'
    else:
        return 'STATIC'


def run_xgb_shap(X, y, label, shap_sample=2000):
    """
    Fit XGBoost with CV, compute SHAP values.
    Returns dict with R², RMSE, SHAP summary DataFrame.
    """
    N, P = X.shape
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)

    # Cross-validated R² and RMSE
    model = xgb.XGBRegressor(**XGB_PARAMS)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=1)
    rmse_scores = cross_val_score(model, X, y, cv=kf,
                                   scoring='neg_root_mean_squared_error', n_jobs=1)

    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    rmse_mean = -rmse_scores.mean()

    print(f"    {label}: R²={r2_mean:.4f} ± {r2_std:.4f}  RMSE={rmse_mean:.4f}  (N={N}, P={P})")

    # Fit on full data for SHAP
    model_full = xgb.XGBRegressor(**XGB_PARAMS)
    model_full.fit(X, y)

    # SHAP values (subsample for speed if large)
    if N > shap_sample:
        X_shap = X.sample(shap_sample, random_state=42)
    else:
        X_shap = X

    explainer = shap.TreeExplainer(model_full)
    shap_values = explainer.shap_values(X_shap)

    # Summary: mean |SHAP| and mean SHAP (for direction)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_shap = shap_values.mean(axis=0)

    shap_df = pd.DataFrame({
        'Variable': X.columns,
        'mean_abs_SHAP': mean_abs_shap,
        'mean_SHAP': mean_shap,
        'Type': [classify_predictor(c) for c in X.columns],
    }).sort_values('mean_abs_SHAP', ascending=False).reset_index(drop=True)

    # Feature importance from XGBoost (for cross-reference)
    shap_df['xgb_importance'] = [model_full.feature_importances_[list(X.columns).index(v)]
                                  for v in shap_df['Variable']]

    return {
        'R2': r2_mean, 'R2_std': r2_std, 'RMSE': rmse_mean,
        'N': N, 'P': P,
        'shap_df': shap_df,
        'shap_values': shap_values,
        'X_shap': X_shap,
        'model': model_full,
    }


# ═══════════════════════════════════════════════════════════════
#  4. EXCLUDE COLUMNS (matching MATLAB)
# ═══════════════════════════════════════════════════════════════
EXCLUDE_EVENT = {'GAGE_ID', 'timescale', 'CLASS', 'ecoregion',
                 'spei_onset', 'spei_termination', 'spei_duration',
                 'ssi_onset', 'ssi_termination', 'ssi_duration',
                 'propagation_lag_months', 'recovery_lag_months',
                 'match_type', 'spei_severity', 'ssi_severity',
                 'spei_rebound_3mo', 'spei_rebound_6mo',
                 'dyn_window_days', 'dyn_zero_flow_fraction',
                 'dyn_recession_k_iqr', 'spei_peak', 'ssi_peak',
                 'BFI_AVE'}


# ═══════════════════════════════════════════════════════════════
#  5. EVENT-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[2] EVENT-LEVEL ANALYSIS")
print("=" * 70)

event_results = []  # collect all rows for summary CSV

for ts in [3, 6]:
    print(f"\n  --- {ts}-MONTH TIMESCALE ---")
    ev = events_m[events_m['timescale'] == ts].copy()

    pred_cols = [c for c in ev.columns if c not in EXCLUDE_EVENT
                 and ev[c].dtype in ['float64', 'int64', 'float32']]
    screened_all = screen_preds(ev[pred_cols])
    screened_ref = [c for c in screened_all if c not in HUMAN_VARS]

    # Define subsets
    subsets = {
        'All':     (ev, screened_all),
        'Ref':     (ev[ev['CLASS'] == 'Ref'], screened_ref),
        'Non-ref': (ev[ev['CLASS'] == 'Non-ref'], screened_all),
    }

    for tgt in ['propagation_lag_months', 'recovery_lag_months']:
        tgt_short = 'prop' if 'prop' in tgt else 'rec'

        for sub_name, (data, preds) in subsets.items():
            clean = data[preds + [tgt]].dropna()
            if len(clean) < 100:
                continue

            label = f"ev_{ts}mo_{tgt_short}_{sub_name}"
            res = run_xgb_shap(clean[preds], clean[tgt], label)

            # Save SHAP summary CSV
            # FORMAT: matches stepwise output (Variable, Beta→mean_abs_SHAP, Coeff→mean_SHAP, Type)
            out_csv = os.path.join(OUT, f"{label}_shap.csv")
            res['shap_df'].to_csv(out_csv, index=False)

            event_results.append({
                'level': 'event', 'timescale': ts, 'target': tgt,
                'target_short': tgt_short, 'subset': sub_name,
                'N': res['N'], 'P': res['P'],
                'R2': res['R2'], 'R2_std': res['R2_std'], 'RMSE': res['RMSE'],
            })

    # ── ECOREGION × CLASS models (for Figure 6) ──
    print(f"\n  --- {ts}-MONTH ECOREGION MODELS ---")
    ecoregions = sorted(ev['ecoregion'].dropna().unique())

    for tgt in ['propagation_lag_months', 'recovery_lag_months']:
        tgt_short = 'prop' if 'prop' in tgt else 'rec'

        for eco in ecoregions:
            for cls, preds in [('Ref', screened_ref), ('Non-ref', screened_all)]:
                mask = (ev['ecoregion'] == eco) & (ev['CLASS'] == cls)
                sub = ev[mask]
                clean = sub[preds + [tgt]].dropna()
                if len(clean) < 50:
                    event_results.append({
                        'level': 'event_eco', 'timescale': ts, 'target': tgt,
                        'target_short': tgt_short, 'subset': f"{eco}_{cls}",
                        'N': len(clean), 'P': len(preds),
                        'R2': np.nan, 'R2_std': np.nan, 'RMSE': np.nan,
                    })
                    continue

                label = f"ev_{ts}mo_{tgt_short}_{eco}_{cls}"

                # Quick fit (no SHAP for ecoregion — just R²)
                model = xgb.XGBRegressor(**XGB_PARAMS)
                kf = KFold(n_splits=min(CV_FOLDS, max(2, len(clean)//30)),
                           shuffle=True, random_state=42)
                try:
                    scores = cross_val_score(model, clean[preds], clean[tgt],
                                              cv=kf, scoring='r2', n_jobs=1)
                    r2 = scores.mean()
                    r2_s = scores.std()
                except:
                    r2, r2_s = np.nan, np.nan

                # Also fit full model for R² (training, for map consistency with MATLAB)
                model.fit(clean[preds], clean[tgt])
                y_pred = model.predict(clean[preds])
                r2_full = r2_score(clean[tgt], y_pred)
                rmse_full = np.sqrt(mean_squared_error(clean[tgt], y_pred))

                print(f"    {label}: CV_R²={r2:.4f}  Full_R²={r2_full:.4f}  N={len(clean)}")

                event_results.append({
                    'level': 'event_eco', 'timescale': ts, 'target': tgt,
                    'target_short': tgt_short, 'subset': f"{eco}_{cls}",
                    'N': len(clean), 'P': len(preds),
                    'R2': r2, 'R2_std': r2_s, 'RMSE': rmse_full,
                    'R2_full': r2_full,
                })


# ═══════════════════════════════════════════════════════════════
#  6. BASIN-LEVEL ANALYSIS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[3] BASIN-LEVEL ANALYSIS")
print("=" * 70)

# Get CLASS for basins from GAGES-II
classif = pd.read_csv(os.path.join(gages_dir, 'conterm_bas_classif.txt'),
                       dtype={'STAID': str}, encoding='latin-1')
classif['GAGE_ID'] = classif['STAID'].str.strip().str.zfill(8)
classif = classif[['GAGE_ID', 'CLASS']]

# Get ecoregion from shapefile data (use events as source)
eco_map = events[['GAGE_ID', 'ecoregion']].drop_duplicates('GAGE_ID')

basin_results = []

for ts in [3, 6]:
    print(f"\n  --- {ts}-MONTH TIMESCALE ---")

    ev_ts = events_m[events_m['timescale'] == ts]
    dyn_basin = ev_ts[['GAGE_ID'] + DYN_COLS].groupby('GAGE_ID').mean().reset_index()
    dyn_basin.columns = ['GAGE_ID'] + [f'mean_{c}' for c in DYN_COLS]

    if ts == 3:
        btgts = {'buffered_3_pct': 'buf', 'independent_3_pct': 'ind',
                 'prop_lag_3_mean': 'plg', 'rec_lag_3_mean': 'rlg'}
    else:
        btgts = {'buffered_6_pct': 'buf', 'independent_6_pct': 'ind',
                 'prop_lag_6_mean': 'plg', 'rec_lag_6_mean': 'rlg'}

    btgts = {k: v for k, v in btgts.items() if k in basins.columns}

    bdf = basins.merge(dyn_basin, on='GAGE_ID', how='inner')
    bdf = bdf.merge(static_df, on='GAGE_ID', how='left')
    bdf = bdf.merge(classif, on='GAGE_ID', how='left')
    bdf = bdf.merge(eco_map, on='GAGE_ID', how='left')

    bpreds_all = [f'mean_{c}' for c in DYN_COLS] + \
                 [c for c in STATIC_COLS if c in bdf.columns and c != 'BFI_AVE']
    screened_b = screen_preds(bdf[bpreds_all])
    screened_b_ref = [c for c in screened_b if c not in HUMAN_VARS]

    print(f"  Basins: {len(bdf)} (Ref:{(bdf['CLASS']=='Ref').sum()}, NR:{(bdf['CLASS']=='Non-ref').sum()})")
    print(f"  Screened preds: All={len(screened_b)}, Ref={len(screened_b_ref)}")

    # ── ALL / REF / NON-REF ──
    for tgt_full, tgt_short in btgts.items():
        subsets_b = {
            'All':     (bdf, screened_b),
            'Ref':     (bdf[bdf['CLASS'] == 'Ref'], screened_b_ref),
            'Non-ref': (bdf[bdf['CLASS'] == 'Non-ref'], screened_b),
        }

        for sub_name, (data, preds) in subsets_b.items():
            ap = [c for c in preds if c in data.columns]
            clean = data[ap + [tgt_full]].dropna()
            if len(clean) < 50:
                continue

            label = f"ba_{ts}mo_{tgt_short}_{sub_name}"
            res = run_xgb_shap(clean[ap], clean[tgt_full], label)

            out_csv = os.path.join(OUT, f"{label}_shap.csv")
            res['shap_df'].to_csv(out_csv, index=False)

            basin_results.append({
                'level': 'basin', 'timescale': ts, 'target': tgt_full,
                'target_short': tgt_short, 'subset': sub_name,
                'N': res['N'], 'P': res['P'],
                'R2': res['R2'], 'R2_std': res['R2_std'], 'RMSE': res['RMSE'],
            })

    # ── ECOREGION × CLASS (for Figure 6 basin-level) ──
    print(f"\n  --- {ts}-MONTH BASIN ECOREGION MODELS ---")
    ecoregions = sorted(bdf['ecoregion'].dropna().unique())

    for tgt_full, tgt_short in btgts.items():
        for eco in ecoregions:
            for cls, preds in [('Ref', screened_b_ref), ('Non-ref', screened_b)]:
                mask = (bdf['ecoregion'] == eco) & (bdf['CLASS'] == cls)
                data = bdf[mask]
                ap = [c for c in preds if c in data.columns]
                clean = data[ap + [tgt_full]].dropna()

                if len(clean) < 30:
                    basin_results.append({
                        'level': 'basin_eco', 'timescale': ts, 'target': tgt_full,
                        'target_short': tgt_short, 'subset': f"{eco}_{cls}",
                        'N': len(clean), 'P': len(ap),
                        'R2': np.nan, 'R2_std': np.nan, 'RMSE': np.nan,
                    })
                    continue

                label = f"ba_{ts}mo_{tgt_short}_{eco}_{cls}"
                model = xgb.XGBRegressor(**XGB_PARAMS)
                n_cv = min(CV_FOLDS, max(2, len(clean) // 20))
                kf = KFold(n_splits=n_cv, shuffle=True, random_state=42)
                try:
                    scores = cross_val_score(model, clean[ap], clean[tgt_full],
                                              cv=kf, scoring='r2', n_jobs=1)
                    r2 = scores.mean()
                except:
                    r2 = np.nan

                # Full-data fit for training R²
                model.fit(clean[ap], clean[tgt_full])
                r2_full = r2_score(clean[tgt_full], model.predict(clean[ap]))
                rmse_full = np.sqrt(mean_squared_error(clean[tgt_full], model.predict(clean[ap])))

                print(f"    {label}: CV_R²={r2:.4f}  Full_R²={r2_full:.4f}  N={len(clean)}")

                basin_results.append({
                    'level': 'basin_eco', 'timescale': ts, 'target': tgt_full,
                    'target_short': tgt_short, 'subset': f"{eco}_{cls}",
                    'N': len(clean), 'P': len(ap),
                    'R2': r2, 'R2_std': np.nan, 'RMSE': rmse_full,
                    'R2_full': r2_full,
                })


# ═══════════════════════════════════════════════════════════════
#  7. EXPORT STRUCTURED RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("[4] EXPORTING RESULTS")
print("=" * 70)

# ── A. Master summary table ──
all_results = event_results + basin_results
summary_df = pd.DataFrame(all_results)
summary_df.to_csv(os.path.join(OUT, 'xgboost_summary_all.csv'), index=False)
print(f"  Saved: xgboost_summary_all.csv ({len(summary_df)} rows)")

# ── B. Figure 4 specific: R² + top 5 predictors for 4 panels ──
# Panel a: buffered_3_pct / basin / All
# Panel b: prop_lag_3_mean / basin / All
# Panel c: rec_lag / event / All (3-month)
# Panel d: independent_3_pct / basin / All
fig4_configs = [
    ('ba_3mo_buf_All', 'buffered_3_pct', 'Panel (a): Attenuated %'),
    ('ba_3mo_plg_All', 'prop_lag_3_mean', 'Panel (b): Propagation Lag'),
    ('ev_3mo_rec_All', 'recovery_lag_months', 'Panel (c): Recovery Lag'),
    ('ba_3mo_ind_All', 'independent_3_pct', 'Panel (d): Independent %'),
]

fig4_rows = []
for label, tgt, panel_name in fig4_configs:
    shap_file = os.path.join(OUT, f"{label}_shap.csv")
    if os.path.exists(shap_file):
        sdf = pd.read_csv(shap_file)
        # Get R² from summary
        match = summary_df[(summary_df['level'].isin(['event','basin'])) &
                           (summary_df['target'].str.contains(tgt.replace('_3_','_3_').split('_')[0][:3])) &
                           (summary_df['subset'] == 'All') &
                           (summary_df['timescale'] == 3)]
        if len(match) == 0:
            # Try exact match
            match = [r for r in all_results if label.replace('ev_3mo_rec_All','') in str(r)]

        r2_val = None
        for r in all_results:
            rlabel = f"{'ev' if r['level']=='event' else 'ba'}_{r['timescale']}mo_{r.get('target_short','')}_{r['subset']}"
            if rlabel == label:
                r2_val = r['R2']
                break

        for _, row in sdf.head(5).iterrows():
            fig4_rows.append({
                'panel': panel_name,
                'label': label,
                'R2': r2_val,
                'rank': _ + 1,
                'Variable': row['Variable'],
                'mean_abs_SHAP': row['mean_abs_SHAP'],
                'mean_SHAP': row['mean_SHAP'],
                'Type': row['Type'],
            })

fig4_df = pd.DataFrame(fig4_rows)
fig4_df.to_csv(os.path.join(OUT, 'figure4_data.csv'), index=False)
print(f"  Saved: figure4_data.csv")

# ── C. Figure 5 specific: Top 8 predictors for 4 panels ──
# Panel a: prop_lag / basin / All (3-mo)
# Panel b: prop_lag / event / All (3-mo)
# Panel c: rec_lag / basin / All (3-mo)
# Panel d: rec_lag / event / All (3-mo)
fig5_configs = [
    ('ba_3mo_plg_All', 'Panel (a): Prop Lag Basin'),
    ('ev_3mo_prop_All', 'Panel (b): Prop Lag Event'),
    ('ba_3mo_rlg_All', 'Panel (c): Rec Lag Basin'),
    ('ev_3mo_rec_All', 'Panel (d): Rec Lag Event'),
]

fig5_rows = []
for label, panel_name in fig5_configs:
    shap_file = os.path.join(OUT, f"{label}_shap.csv")
    if os.path.exists(shap_file):
        sdf = pd.read_csv(shap_file)
        for i, row in sdf.head(8).iterrows():
            fig5_rows.append({
                'panel': panel_name,
                'label': label,
                'rank': i + 1,
                'Variable': row['Variable'],
                'mean_abs_SHAP': row['mean_abs_SHAP'],
                'mean_SHAP': row['mean_SHAP'],
                'Type': row['Type'],
            })

fig5_df = pd.DataFrame(fig5_rows)
fig5_df.to_csv(os.path.join(OUT, 'figure5_data.csv'), index=False)
print(f"  Saved: figure5_data.csv")

# ── D. Figure 6 specific: Ecoregion R² for maps ──
fig6_rows = []
for r in all_results:
    if r['level'] in ('event_eco', 'basin_eco'):
        sub = r['subset']
        if '_' in sub:
            eco, cls = sub.rsplit('_', 1)
            fig6_rows.append({
                'level': r['level'].replace('_eco', ''),
                'timescale': r['timescale'],
                'target': r['target'],
                'target_short': r['target_short'],
                'ecoregion': eco,
                'CLASS': cls,
                'N': r['N'],
                'R2_cv': r['R2'],
                'R2_full': r.get('R2_full', np.nan),
                'RMSE': r['RMSE'],
            })

fig6_df = pd.DataFrame(fig6_rows)
fig6_df.to_csv(os.path.join(OUT, 'figure6_ecoregion_R2.csv'), index=False)
print(f"  Saved: figure6_ecoregion_R2.csv ({len(fig6_df)} rows)")

# ── E. Figure 7 specific: Storage duality (dyn_BFI SHAP by ecoregion) ──
# Need dyn_BFI SHAP direction per ecoregion for prop and rec
# Also Ref vs Non-ref R² comparison
print("\n  Computing Figure 7 storage duality SHAP...")

fig7_duality = []
fig7_refnr = []

for ts in [3, 6]:
    ev = events_m[events_m['timescale'] == ts].copy()
    pred_cols = [c for c in ev.columns if c not in EXCLUDE_EVENT
                 and ev[c].dtype in ['float64', 'int64', 'float32']]
    screened_all = screen_preds(ev[pred_cols])

    ecoregions = sorted(ev['ecoregion'].dropna().unique())

    for tgt in ['propagation_lag_months', 'recovery_lag_months']:
        tgt_short = 'prop' if 'prop' in tgt else 'rec'

        for eco in ecoregions:
            for cls in ['Ref', 'Non-ref']:
                preds = [c for c in screened_all if c not in HUMAN_VARS] if cls == 'Ref' else screened_all
                mask = (ev['ecoregion'] == eco) & (ev['CLASS'] == cls)
                clean = ev[mask][preds + [tgt]].dropna()

                if len(clean) < 50 or 'dyn_BFI' not in preds:
                    fig7_duality.append({
                        'timescale': ts, 'target': tgt_short,
                        'ecoregion': eco, 'CLASS': cls,
                        'dyn_BFI_mean_SHAP': np.nan, 'N': len(clean),
                    })
                    continue

                model = xgb.XGBRegressor(**XGB_PARAMS)
                model.fit(clean[preds], clean[tgt])

                # SHAP for this subset
                X_shap = clean[preds].sample(min(len(clean), 1000), random_state=42)
                explainer = shap.TreeExplainer(model)
                sv = explainer.shap_values(X_shap)

                bfi_idx = list(preds).index('dyn_BFI')
                bfi_mean_shap = sv[:, bfi_idx].mean()
                bfi_abs_shap = np.abs(sv[:, bfi_idx]).mean()

                fig7_duality.append({
                    'timescale': ts, 'target': tgt_short,
                    'ecoregion': eco, 'CLASS': cls,
                    'dyn_BFI_mean_SHAP': bfi_mean_shap,
                    'dyn_BFI_abs_SHAP': bfi_abs_shap,
                    'N': len(clean),
                })

                print(f"    {ts}mo {tgt_short} {eco} {cls}: dyn_BFI SHAP={bfi_mean_shap:+.4f} (N={len(clean)})")

# Ref vs Non-ref R² comparison (for Fig 7 panels c,d)
for r in all_results:
    if r['level'] in ('event',) and r['subset'] in ('Ref', 'Non-ref'):
        fig7_refnr.append({
            'timescale': r['timescale'],
            'target': r['target'],
            'target_short': r['target_short'],
            'CLASS': r['subset'],
            'R2': r['R2'],
            'R2_std': r['R2_std'],
            'N': r['N'],
        })

fig7_dual_df = pd.DataFrame(fig7_duality)
fig7_dual_df.to_csv(os.path.join(OUT, 'figure7_dyn_BFI_shap.csv'), index=False)
print(f"  Saved: figure7_dyn_BFI_shap.csv ({len(fig7_dual_df)} rows)")

fig7_rn_df = pd.DataFrame(fig7_refnr)
fig7_rn_df.to_csv(os.path.join(OUT, 'figure7_ref_nonref_R2.csv'), index=False)
print(f"  Saved: figure7_ref_nonref_R2.csv ({len(fig7_rn_df)} rows)")


# ═══════════════════════════════════════════════════════════════
#  8. PRINT GRAND SUMMARY + HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GRAND SUMMARY: MAIN CONFIGURATIONS")
print("=" * 70)

main = summary_df[summary_df['level'].isin(['event', 'basin'])].copy()
print(f"\n{'Config':<35s} {'N':>6s} {'R²':>8s} {'±':>6s} {'RMSE':>8s}")
print("-" * 70)
for _, r in main.iterrows():
    label = f"{r['level']}_{r['timescale']}mo_{r['target_short']}_{r['subset']}"
    print(f"{label:<35s} {r['N']:>6.0f} {r['R2']:>8.4f} {r['R2_std']:>6.4f} {r['RMSE']:>8.4f}")

elapsed = time.time() - t_global
print(f"\n{'='*70}")
print(f"HYPERPARAMETERS (XGBoost)")
print(f"{'='*70}")
print(f"  n_estimators:     {XGB_PARAMS['n_estimators']}")
print(f"  max_depth:        {XGB_PARAMS['max_depth']}")
print(f"  learning_rate:    {XGB_PARAMS['learning_rate']}")
print(f"  subsample:        {XGB_PARAMS['subsample']}")
print(f"  colsample_bytree: {XGB_PARAMS['colsample_bytree']}")
print(f"  min_child_weight: {XGB_PARAMS['min_child_weight']}")
print(f"  reg_alpha (L1):   {XGB_PARAMS['reg_alpha']}")
print(f"  reg_lambda (L2):  {XGB_PARAMS['reg_lambda']}")
print(f"  CV folds:         {CV_FOLDS}")
print(f"  SHAP method:      TreeExplainer (exact)")
print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

# Save hyperparams to JSON
with open(os.path.join(OUT, 'hyperparameters.json'), 'w') as f:
    json.dump({**XGB_PARAMS, 'cv_folds': CV_FOLDS, 'shap_method': 'TreeExplainer'}, f, indent=2)

print(f"\n{'='*70}")
print("OUTPUT FILES (in XGBoost_Results/):")
print("  xgboost_summary_all.csv        — Master table of all R²/RMSE")
print("  figure4_data.csv               — Top 5 SHAP + R² for 4 panels")
print("  figure5_data.csv               — Top 8 SHAP rankings for 4 panels")
print("  figure6_ecoregion_R2.csv       — Ecoregion×CLASS R² for maps")
print("  figure7_dyn_BFI_shap.csv       — dyn_BFI SHAP by eco (storage duality)")
print("  figure7_ref_nonref_R2.csv      — Ref vs Non-ref R²")
print("  *_shap.csv                     — Per-config full SHAP tables")
print("  hyperparameters.json           — Model configuration")
print("=" * 70)
print("DONE!")
