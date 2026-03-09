"""
Step 1: Drought Event Matching Analysis — 4-Step Protocol
==========================================================
Precise implementation of the 4-step matching protocol for linking
meteorological (SPEI) and hydrological (SSI) drought events.

Step 1: For each SSI drought, search for SPEI droughts whose active period
        (onset to termination) overlaps with a window from [SSI_onset - 6 months]
        to [SSI_onset]. If multiple SPEI events fall within the window, select
        the one with the largest temporal overlap WITH THE SSI EVENT.

Step 2: Only retain pairs where SPEI onset <= SSI onset (propagation lag >= 0).
        Pairs where SSI leads SPEI are separated into a distinct category.

Step 3: Exclude SPEI events with cumulative severity weaker than -3.0.

Step 4: Classify unmatched events as buffered (SPEI) or independent (SSI).

Drought detection rules (Yevjevich 1967, run theory with pooling):
- Onset: 3 consecutive months <= -1.0
- Continuation: allows 1 interruption where -1.0 < val <= -0.5
- Termination: 2 consecutive months > -0.5
- Termination date = last day of termination month
- False recoveries tracked per event

Inputs:
  - tier1_3606_SPEI_SSI_timeseries.csv  (monthly time series for 3,606 CONUS basins)

Outputs:
  - drought_matched_pairs.csv          (matched SPEI-SSI pairs)
  - drought_basin_classification.csv   (per-basin summary stats)
  - drought_ssi_leads_pairs.csv        (SSI-leads-SPEI pairs)
  - drought_buffered_events.csv        (buffered SPEI events)
  - drought_independent_events.csv     (independently initiated SSI events)
"""

import pandas as pd
import numpy as np
from calendar import monthrange
import time
import os
from pathlib import Path


# ============================================================
# PATH CONFIGURATION — relative to this script
# ============================================================
STEP_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
RESULTS_ROOT = STEP_DIR.parent
DATA_ROOT = RESULTS_ROOT.parent


# ============================================================
# DROUGHT DETECTION
# ============================================================

def last_day_of_month(date_str):
    """Return last day of the month for a YYYY-MM-DD date string."""
    dt = pd.to_datetime(date_str)
    day = monthrange(dt.year, dt.month)[1]
    return f"{dt.year}-{dt.month:02d}-{day:02d}"


def detect_droughts(values, dates, onset_thresh=-1.0, term_thresh=-0.5):
    """
    Detect drought events using run theory with pooling.

    Onset: 3 consecutive months <= onset_thresh
    Continuation: allows 1 month in (onset_thresh, term_thresh] zone
    Termination: 2 consecutive months > term_thresh
    Termination date = last day of the last drought month
    """
    n = len(values)
    droughts = []
    i = 0

    while i < n:
        # Look for onset: 3 consecutive months <= onset_thresh
        if (i + 2 < n
            and values[i] is not None and values[i] <= onset_thresh
            and values[i+1] is not None and values[i+1] <= onset_thresh
            and values[i+2] is not None and values[i+2] <= onset_thresh):

            onset_idx = i
            false_recoveries = 0

            # Scan forward to find termination
            j = i + 3
            while j < n:
                v = values[j]

                if v is None:
                    j += 1
                    continue

                if v <= onset_thresh:
                    j += 1
                    continue

                if v > onset_thresh and v <= term_thresh:
                    # Interruption zone: allow exactly 1 consecutive month
                    if j + 1 < n and values[j+1] is not None and values[j+1] <= onset_thresh:
                        false_recoveries += 1
                        j += 2
                        continue
                    else:
                        if j + 1 < n and values[j+1] is not None and values[j+1] > term_thresh:
                            break
                        elif j + 1 < n and values[j+1] is not None and values[j+1] <= onset_thresh:
                            false_recoveries += 1
                            j += 2
                            continue
                        else:
                            false_recoveries += 1
                            j += 1
                            continue

                if v > term_thresh:
                    if j + 1 < n and values[j+1] is not None and values[j+1] > term_thresh:
                        break
                    else:
                        false_recoveries += 1
                        j += 1
                        continue

                j += 1

            # Drought ended: last drought month is j-1
            if j >= n:
                end_idx = n - 1
            else:
                end_idx = j - 1

            # Calculate event properties
            duration = end_idx - onset_idx + 1
            event_vals = [values[k] for k in range(onset_idx, end_idx + 1)
                         if values[k] is not None]

            if len(event_vals) == 0 or duration < 3:
                i = end_idx + 1
                continue

            severity = sum(event_vals)
            peak_value = min(event_vals)
            intensity = severity / duration

            # Find peak date
            peak_date = dates[onset_idx]
            for k in range(onset_idx, end_idx + 1):
                if values[k] is not None and values[k] == peak_value:
                    peak_date = dates[k]
                    break

            droughts.append({
                'onset': dates[onset_idx],
                'onset_idx': onset_idx,
                'termination': last_day_of_month(dates[end_idx]),
                'term_idx': end_idx,
                'duration_months': duration,
                'peak_value': round(peak_value, 4),
                'peak_date': peak_date,
                'severity': round(severity, 4),
                'intensity': round(intensity, 4),
                'false_recoveries': false_recoveries
            })

            i = end_idx + 1
        else:
            i += 1

    return droughts


# ============================================================
# 4-STEP MATCHING PROTOCOL
# ============================================================

def match_events(spei_droughts, ssi_droughts, window_months=6, severity_cutoff=-3.0):
    """
    Precisely follows the 4-step matching protocol.

    Returns:
        matched_pairs:   causal matches (SPEI onset <= SSI onset)
        ssi_leads_pairs: pairs where SSI onset < SPEI onset (separate category)
        buffered_spei:   SPEI droughts with no matching SSI
        independent_ssi: SSI droughts with no SPEI candidate at all
        n_unique_spei_matched: count of unique SPEI events that got matched
        n_ssi_leads: count of SSI events in SSI-leads category
    """
    matched_pairs = []
    ssi_leads_pairs = []
    matched_ssi_indices = set()
    matched_spei_indices = set()

    for ssi_idx, ssi in enumerate(ssi_droughts):
        ssi_onset = ssi['onset_idx']
        ssi_term = ssi['term_idx']

        # === STEP 1: Search window ===
        # Window = [SSI_onset - window_months, SSI_onset]
        win_start = ssi_onset - window_months
        win_end = ssi_onset

        # Find all SPEI droughts whose active period overlaps with this window
        candidates = []

        for spei_idx, spei in enumerate(spei_droughts):
            spei_onset = spei['onset_idx']
            spei_term = spei['term_idx']

            # === STEP 3: Severity filter ===
            # Exclude SPEI with cumulative severity weaker than cutoff
            # severity is negative; > cutoff means weaker (e.g., -2.0 > -3.0)
            if spei['severity'] > severity_cutoff:
                continue

            # Check overlap between SPEI active period and search window
            if spei_term < win_start or spei_onset > win_end:
                continue

            # This SPEI is a candidate - compute overlap WITH THE SSI EVENT
            ovl_start = max(spei_onset, ssi_onset)
            ovl_end = min(spei_term, ssi_term)
            overlap_with_ssi = max(0, ovl_end - ovl_start + 1)

            # Propagation lag = SSI_onset - SPEI_onset
            prop_lag = ssi_onset - spei_onset

            candidates.append({
                'spei_idx': spei_idx,
                'spei': spei,
                'overlap_with_ssi': overlap_with_ssi,
                'prop_lag': prop_lag
            })

        if not candidates:
            continue

        # === STEP 2: Directionality filter ===
        causal = [c for c in candidates if c['prop_lag'] >= 0]
        ssi_leads = [c for c in candidates if c['prop_lag'] < 0]

        if causal:
            # Select SPEI with largest temporal overlap with SSI event
            best = max(causal, key=lambda c: (c['overlap_with_ssi'], -c['prop_lag']))

            spei = best['spei']
            prop_lag = best['prop_lag']
            rec_lag = ssi['term_idx'] - spei['term_idx']

            matched_pairs.append({
                'spei_onset': spei['onset'],
                'spei_termination': spei['termination'],
                'spei_duration': spei['duration_months'],
                'spei_severity': spei['severity'],
                'spei_peak': spei['peak_value'],
                'spei_intensity': spei['intensity'],
                'spei_false_rec': spei['false_recoveries'],
                'ssi_onset': ssi['onset'],
                'ssi_termination': ssi['termination'],
                'ssi_duration': ssi['duration_months'],
                'ssi_severity': ssi['severity'],
                'ssi_peak': ssi['peak_value'],
                'ssi_intensity': ssi['intensity'],
                'ssi_false_rec': ssi['false_recoveries'],
                'propagation_lag_months': prop_lag,
                'recovery_lag_months': rec_lag,
            })
            matched_ssi_indices.add(ssi_idx)
            matched_spei_indices.add(best['spei_idx'])

        elif ssi_leads:
            best = max(ssi_leads, key=lambda c: c['overlap_with_ssi'])
            spei = best['spei']

            ssi_leads_pairs.append({
                'spei_onset': spei['onset'],
                'spei_termination': spei['termination'],
                'spei_duration': spei['duration_months'],
                'spei_severity': spei['severity'],
                'ssi_onset': ssi['onset'],
                'ssi_termination': ssi['termination'],
                'ssi_duration': ssi['duration_months'],
                'ssi_severity': ssi['severity'],
                'lag_months': best['prop_lag'],
            })

    # === STEP 4: Classify unmatched ===
    # Buffered SPEI: all SPEI not in any causal match
    buffered_spei = []
    for spei_idx, spei in enumerate(spei_droughts):
        if spei_idx not in matched_spei_indices:
            if spei['severity'] > severity_cutoff:
                reason = 'severity_below_threshold'
            else:
                reason = 'no_matching_ssi'
            buffered_spei.append({**spei, 'reason': reason})

    # Independent SSI: SSI with no candidate at all (not matched, not ssi-leads)
    ssi_leads_indices = set()
    for ssi_idx, ssi in enumerate(ssi_droughts):
        if ssi_idx not in matched_ssi_indices:
            ssi_onset = ssi['onset_idx']
            win_start = ssi_onset - window_months
            win_end = ssi_onset
            had_candidate = False
            for spei in spei_droughts:
                if spei['severity'] > severity_cutoff:
                    continue
                if spei['term_idx'] < win_start or spei['onset_idx'] > win_end:
                    continue
                had_candidate = True
                break
            if had_candidate:
                ssi_leads_indices.add(ssi_idx)

    independent_ssi = []
    for ssi_idx, ssi in enumerate(ssi_droughts):
        if ssi_idx not in matched_ssi_indices and ssi_idx not in ssi_leads_indices:
            independent_ssi.append(ssi)

    return (matched_pairs, ssi_leads_pairs, buffered_spei, independent_ssi,
            len(matched_spei_indices), len(ssi_leads_indices))


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("STEP 1: DROUGHT EVENT MATCHING — 4-STEP PROTOCOL")
    print("=" * 60)

    csv_path = str(DATA_ROOT / "tier1_3606_SPEI_SSI_timeseries.csv")
    out_dir = str(STEP_DIR)

    print(f"\nReading CSV from {csv_path}...")
    t0 = time.time()
    df = pd.read_csv(csv_path, dtype={'GAGE_ID': str})
    print(f"  {len(df)} rows, {df['GAGE_ID'].nunique()} gauges, {time.time()-t0:.1f}s")

    gauge_ids = df['GAGE_ID'].unique()

    all_basin_stats = []
    all_matched = []
    all_ssi_leads = []
    all_buffered = []
    all_independent = []

    timescales = [('3', 'SPEI_3', 'SSI_3'), ('6', 'SPEI_6', 'SSI_6')]

    print("\nProcessing gauges...")
    t0 = time.time()

    for g_num, gid in enumerate(gauge_ids):
        if (g_num + 1) % 500 == 0:
            print(f"  {g_num+1}/{len(gauge_ids)} ({time.time()-t0:.0f}s)")

        gdf = df[df['GAGE_ID'] == gid].sort_values('date').reset_index(drop=True)
        dates = gdf['date'].tolist()

        basin = {'GAGE_ID': gid}

        for ts, spei_col, ssi_col in timescales:
            spei_vals = [None if pd.isna(v) else v for v in gdf[spei_col].tolist()]
            ssi_vals = [None if pd.isna(v) else v for v in gdf[ssi_col].tolist()]

            spei_droughts = detect_droughts(spei_vals, dates)
            ssi_droughts = detect_droughts(ssi_vals, dates)

            (matched, ssi_leads, buffered, independent,
             n_unique_spei, n_ssi_leads) = match_events(
                spei_droughts, ssi_droughts,
                window_months=6, severity_cutoff=-3.0
            )

            total_spei = len(spei_droughts)
            total_ssi = len(ssi_droughts)
            n_matched = len(matched)
            n_buffered = len(buffered)
            n_independent = len(independent)
            n_ssi_leads_count = n_ssi_leads

            basin[f'SPEI_{ts}_total'] = total_spei
            basin[f'SSI_{ts}_total'] = total_ssi
            basin[f'matched_pairs_{ts}'] = n_matched
            basin[f'unique_spei_matched_{ts}'] = n_unique_spei
            basin[f'buffered_{ts}_count'] = n_buffered
            basin[f'ssi_leads_{ts}_count'] = n_ssi_leads_count
            basin[f'independent_{ts}_count'] = n_independent
            basin[f'buffered_{ts}_pct'] = round(100 * n_buffered / total_spei, 1) if total_spei > 0 else 0
            basin[f'independent_{ts}_pct'] = round(100 * n_independent / total_ssi, 1) if total_ssi > 0 else 0
            basin[f'ssi_leads_{ts}_pct'] = round(100 * n_ssi_leads_count / total_ssi, 1) if total_ssi > 0 else 0
            basin[f'spei_matched_{ts}_pct'] = round(100 * n_unique_spei / total_spei, 1) if total_spei > 0 else 0
            basin[f'ssi_matched_{ts}_pct'] = round(100 * n_matched / total_ssi, 1) if total_ssi > 0 else 0

            if n_matched > 0:
                plags = [m['propagation_lag_months'] for m in matched]
                rlags = [m['recovery_lag_months'] for m in matched]
                basin[f'prop_lag_{ts}_mean'] = round(np.mean(plags), 2)
                basin[f'prop_lag_{ts}_median'] = round(np.median(plags), 2)
                basin[f'rec_lag_{ts}_mean'] = round(np.mean(rlags), 2)
                basin[f'rec_lag_{ts}_median'] = round(np.median(rlags), 2)
            else:
                basin[f'prop_lag_{ts}_mean'] = None
                basin[f'prop_lag_{ts}_median'] = None
                basin[f'rec_lag_{ts}_mean'] = None
                basin[f'rec_lag_{ts}_median'] = None

            for m in matched:
                all_matched.append({'GAGE_ID': gid, 'timescale': ts, **m})

            for sl in ssi_leads:
                all_ssi_leads.append({'GAGE_ID': gid, 'timescale': ts, **sl})

            for b in buffered:
                all_buffered.append({
                    'GAGE_ID': gid, 'timescale': ts, 'type': 'buffered_SPEI',
                    'onset': b['onset'], 'termination': b['termination'],
                    'duration_months': b['duration_months'], 'severity': b['severity'],
                    'peak_value': b['peak_value'], 'intensity': b['intensity'],
                    'false_recoveries': b['false_recoveries'], 'reason': b['reason'],
                })

            for ind in independent:
                all_independent.append({
                    'GAGE_ID': gid, 'timescale': ts, 'type': 'independent_SSI',
                    'onset': ind['onset'], 'termination': ind['termination'],
                    'duration_months': ind['duration_months'], 'severity': ind['severity'],
                    'peak_value': ind['peak_value'], 'intensity': ind['intensity'],
                    'false_recoveries': ind['false_recoveries'],
                })

        all_basin_stats.append(basin)

    elapsed = time.time() - t0
    print(f"\n  Done: {len(gauge_ids)} gauges in {elapsed:.0f}s")

    # === WRITE CSVs ===
    print("\nWriting output files...")

    df_basin = pd.DataFrame(all_basin_stats)
    p1 = os.path.join(out_dir, "drought_basin_classification.csv")
    df_basin.to_csv(p1, index=False)
    print(f"  {p1} ({len(df_basin)} basins, {len(df_basin.columns)} cols)")

    df_matched = pd.DataFrame(all_matched)
    p2 = os.path.join(out_dir, "drought_matched_pairs.csv")
    df_matched.to_csv(p2, index=False)
    print(f"  {p2} ({len(df_matched)} matched pairs)")

    df_ssi_leads = pd.DataFrame(all_ssi_leads)
    p3 = os.path.join(out_dir, "drought_ssi_leads_pairs.csv")
    df_ssi_leads.to_csv(p3, index=False)
    print(f"  {p3} ({len(df_ssi_leads)} ssi-leads pairs)")

    df_buffered = pd.DataFrame(all_buffered)
    p4 = os.path.join(out_dir, "drought_buffered_events.csv")
    df_buffered.to_csv(p4, index=False)
    print(f"  {p4} ({len(df_buffered)} buffered SPEI)")

    df_independent = pd.DataFrame(all_independent)
    p5 = os.path.join(out_dir, "drought_independent_events.csv")
    df_independent.to_csv(p5, index=False)
    print(f"  {p5} ({len(df_independent)} independent SSI)")

    # === VALIDATION ===
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    for ts in ['3', '6']:
        print(f"\n--- {ts}-month timescale ---")
        total_spei = df_basin[f'SPEI_{ts}_total'].sum()
        total_ssi = df_basin[f'SSI_{ts}_total'].sum()
        n_matched = df_basin[f'matched_pairs_{ts}'].sum()
        n_unique_spei = df_basin[f'unique_spei_matched_{ts}'].sum()
        n_buffered = df_basin[f'buffered_{ts}_count'].sum()
        n_ssi_leads = df_basin[f'ssi_leads_{ts}_count'].sum()
        n_independent = df_basin[f'independent_{ts}_count'].sum()

        print(f"  SPEI total:       {total_spei}")
        print(f"  SSI total:        {total_ssi}")
        print(f"  Matched pairs:    {n_matched} ({100*n_matched/total_ssi:.1f}% of SSI)")
        print(f"  SSI-leads pairs:  {n_ssi_leads} ({100*n_ssi_leads/total_ssi:.1f}% of SSI)")
        print(f"  Independent SSI:  {n_independent} ({100*n_independent/total_ssi:.1f}% of SSI)")
        print(f"  Buffered SPEI:    {n_buffered} ({100*n_buffered/total_spei:.1f}% of SPEI)")
        print(f"  Unique SPEI matched: {n_unique_spei} ({100*n_unique_spei/total_spei:.1f}% of SPEI)")

        # Check accounting
        check1 = n_unique_spei + n_buffered == total_spei
        check2 = n_matched + n_ssi_leads + n_independent == total_ssi
        print(f"  CHECK unique_spei + buffered = total_spei: {n_unique_spei}+{n_buffered}={n_unique_spei+n_buffered} == {total_spei}: {check1}")
        print(f"  CHECK matched + ssi_leads + independent = total_ssi: {n_matched}+{n_ssi_leads}+{n_independent}={n_matched+n_ssi_leads+n_independent} == {total_ssi}: {check2}")

        if len(df_matched[df_matched['timescale'] == ts]) > 0:
            sub = df_matched[df_matched['timescale'] == ts]
            print(f"  Propagation lag: min={sub['propagation_lag_months'].min()}, "
                  f"max={sub['propagation_lag_months'].max()}, "
                  f"mean={sub['propagation_lag_months'].mean():.2f}, "
                  f"median={sub['propagation_lag_months'].median():.1f}")
            print(f"  Recovery lag:    min={sub['recovery_lag_months'].min()}, "
                  f"max={sub['recovery_lag_months'].max()}, "
                  f"mean={sub['recovery_lag_months'].mean():.2f}, "
                  f"median={sub['recovery_lag_months'].median():.1f}")
            neg_prop = (sub['propagation_lag_months'] < 0).sum()
            neg_rec = (sub['recovery_lag_months'] < 0).sum()
            print(f"  Negative prop lags: {neg_prop} (must be 0)")
            print(f"  Negative rec lags:  {neg_rec}")

    print(f"\n{'=' * 60}")
    print(f"TOTAL MATCHED PAIRS: {len(df_matched)}")
    print(f"{'=' * 60}")
    print("\nDone!")


if __name__ == '__main__':
    main()
