# Hydrological Drought Propagation is Memory-driven but Recovery is Event-driven

[![DOI](https://zenodo.org/badge/DOI/PLACEHOLDER.svg)](https://doi.org/PLACEHOLDER)

**Authors:** Amobichukwu C. Amanambu et al.

**Journal:** *Nature Communications* (under review)

## Overview

This repository contains analysis code and derived datasets for studying
hydrological drought propagation and recovery across 3,606 CONUS basins
(1980–2025). We use SPEI and SSI drought indices, event-level matching,
dynamic catchment memory metrics (baseflow index, recession constant,
discharge coefficient of variation), XGBoost/SHAP attribution, and stepwise
regression to demonstrate that drought propagation is controlled by catchment
memory while recovery is driven by event-specific climate forcing.

## Repository Structure

```
├── 01_compute_SPEI.py            # SPEI computation (GLO distribution + L-moments)
├── 02_compute_SSI.py             # SSI computation with QC and aridity filtering
├── 03_merge_SPEI_SSI.py          # Merge SPEI and SSI for Tier 1 basins
├── 04_event_matching.py          # Drought event detection and SPEI–SSI matching
├── 05_recovery_atlas.py          # Recovery lag and half-life computation
├── 06_memory_covariates.py       # Dynamic BFI, recession k, discharge CV
├── 07_attribution_prediction.py  # Shapley variance partitioning and Random Forest
├── 08_compute_dynamic_bfi.py     # Dynamic BFI for buffered/independent events
├── 09_summary_tables.py          # Formatted summary workbook generation
├── 10_xgboost_shap_analysis.py   # XGBoost + SHAP feature attribution
├── 11_event_level_stepwise.m     # MATLAB: event-level stepwise regression
├── 12_basin_level_stepwise.m     # MATLAB: basin-level stepwise regression
├── Data/                         # Derived datasets (small files; see below)
└── README.txt                    # Detailed script-by-script documentation
```

## Pipeline

The scripts are numbered in execution order and organised into three stages:

| Stage | Scripts | Description |
|-------|---------|-------------|
| Index Computation | 01–03 | Compute SPEI and SSI from raw climate and discharge data, merge into a single Tier 1 time-series file |
| Data Preparation | 04–09 | Event matching, recovery metrics, dynamic catchment memory covariates, variance partitioning, and summary tables |
| Analysis | 10–12 | XGBoost/SHAP attribution (Python) and forward stepwise regression (MATLAB) at event and basin levels |

## Data Availability

**Included in this repository** (`Data/`):

| File | Size | Description |
|------|------|-------------|
| `drought_matched_pairs.csv` | 8.3 MB | Matched meteorological–hydrological drought event pairs |
| `drought_basin_classification.csv` | 404 KB | Basin-level summary statistics per timescale |

**Archived on Zenodo** (large files):

| File | Size | Description |
|------|------|-------------|
| `tier1_3606_SPEI_SSI_timeseries.csv` | 156 MB | Monthly SPEI and SSI for 3,606 basins (1980–2025) |
| `all_events_with_dynamic_bfi.csv` | 47 MB | All drought events with dynamic BFI and recession metrics |
| `matched_pairs_dynamic.csv` | 13 MB | Matched pairs with dynamic catchment memory metrics |

**External data sources** (publicly available):

- USGS daily streamflow via [NWIS](https://waterdata.usgs.gov/nwis)
- gridMET precipitation and PET ([Abatzoglou, 2013](https://doi.org/10.1002/joc.3413))
- GAGES-II basin attributes and shapefiles ([Falcone, 2011](https://doi.org/10.3133/sir20115263))

## Software Requirements

- **Python** >= 3.9 with: `numpy`, `pandas`, `scipy`, `scikit-learn`, `xgboost`, `shap`, `statsmodels`, `openpyxl`
- **MATLAB** R2020b or later (for scripts 11–12)

## How to Reproduce

1. Download external data sources listed above.
2. Download the large derived datasets from Zenodo and place them in `Data/`.
3. Run scripts 01–03 to compute drought indices (or use the pre-computed `tier1_3606_SPEI_SSI_timeseries.csv`).
4. Run scripts 04–09 sequentially for data preparation.
5. Run scripts 10–12 for the final analysis.

See `README.txt` for detailed per-script documentation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use this code or data, please cite:

> Amanambu, A.C. et al. Hydrological Drought Propagation is Memory-driven but Recovery is Event-driven. *Nature Communications* (under review).
