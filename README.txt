Code and Data Availability
==========================

Hydrological Drought Propagation is Memory-driven but Recovery is Event-driven

All analysis scripts are numbered in pipeline execution order.
Scripts should be run from this folder; paths resolve to the parent directory
where Data/, gagesii_csv/, XGBoost_Results/, and Manuscript/ reside.

Derived Datasets (Data/)
------------------------
The Data/ subfolder contains the derived datasets produced by the analysis
pipeline. These are provided so that users can reproduce results from any
entry point without reprocessing raw data.

tier1_3606_SPEI_SSI_timeseries.csv   Monthly SPEI and SSI time series
  (156 MB)                           (3- and 6-month accumulations) for
                                     3,606 non-arid Tier 1 CONUS basins
                                     (1980-2025). Input to Script 04.

drought_matched_pairs.csv            Matched meteorological-hydrological
  (8.3 MB)                           drought event pairs with propagation
                                     and recovery lags. Output of Script 04.

all_events_with_dynamic_bfi.csv      All drought events (propagated, buffered,
  (47 MB)                            independent) with event-level dynamic BFI,
                                     recession constant, and discharge CV.
                                     Input to Scripts 09-10.

matched_pairs_dynamic.csv            Matched pairs augmented with dynamic
  (13 MB)                            catchment memory metrics and static basin
                                     attributes. Input to Scripts 10-12.

drought_basin_classification.csv     Basin-level summary statistics per
  (404 KB)                           timescale (event counts, median lags,
                                     attenuation and independence fractions).
                                     Input to Scripts 10, 12.

Index Computation (01-03)
-------------------------
01_compute_SPEI.py               Compute SPEI using the gold standard methodology
                                 (Vicente-Serrano et al., 2010). Three-parameter
                                 Generalized Logistic distribution with L-moments.
                                 Accumulations at 3- and 6-month timescales.

02_compute_SSI.py                Compute SSI with integrated quality control and
                                 aridity filtering. Reads USGS daily discharge,
                                 screens for Tier 1 data quality (>=90% complete,
                                 >=30 yr), fits Gamma distribution with L-moments
                                 per calendar month, and removes persistently arid
                                 basins (>50% drought frequency).

03_merge_SPEI_SSI.py             Merge SPEI and SSI into a single Tier 1 time-series
                                 file (SPEI_3, SPEI_6, SSI_3, SSI_6) for 3,606
                                 non-arid CONUS basins.

Data Preparation Pipeline (04-09)
---------------------------------
04_event_matching.py             Drought event detection and matching. Identifies
                                 meteorological (SPEI) and hydrological (SSI) drought
                                 events across 3,606 basins using threshold-based run
                                 theory, then pairs them into matched drought couplets.
                                 Classifies unmatched events as buffered or independently
                                 initiated.

05_recovery_atlas.py             Recovery metric computation. Calculates recovery lag,
                                 half-life, and false recovery count for each matched
                                 pair using a 36-month search window. Adds antecedent
                                 SSI and SPEI rebound metrics.

06_memory_covariates.py          Dynamic catchment memory and covariate analysis.
                                 Computes event-specific BFI, recession constant, and
                                 discharge CV from a 3-year trailing window of daily
                                 discharge before each event onset. Runs Spearman
                                 correlations and Random Forest importance analysis.

07_attribution_prediction.py     Hierarchical variance partitioning across five predictor
                                 groups using Shapley decomposition (120 OLS permutations).
                                 Builds Random Forest models and classifies basins into
                                 recovery regimes (climate-controlled, memory-controlled,
                                 mixed).

08_compute_dynamic_bfi.py        Extends dynamic BFI and recession metric computation to
                                 buffered and independently initiated events not included
                                 in matched pairs.

09_summary_tables.py             Generates formatted Excel summary workbook of stepwise
                                 regression results across all four outcomes with
                                 ecoregion detail and key findings.

Analysis Pipeline (10-12)
-------------------------
10_xgboost_shap_analysis.py      XGBoost gradient boosting with SHAP-based feature
                                 attribution. Predicts propagation lag and recovery lag
                                 at basin and event levels, stratified by ecoregion and
                                 reference/non-reference classification.

11_event_level_stepwise.m        Event-level forward stepwise regression (MATLAB).
                                 Predicts propagation and recovery lag using dynamic +
                                 static predictors across ecoregion and class subsets.

12_basin_level_stepwise.m        Basin-level forward stepwise regression (MATLAB) with
                                 aggregated dynamic variables.

Software Requirements
---------------------
Python >= 3.9 with: numpy, pandas, scipy, scikit-learn, xgboost, shap
MATLAB R2020b or later (for scripts 11-12)

External Data Sources
---------------------
The following publicly available datasets are required for Scripts 01-02
(index computation) and 05-06 (basin attribute merging):

- USGS daily discharge via NWIS (https://waterdata.usgs.gov/nwis)
- gridMET precipitation and PET (Abatzoglou, 2013)
- GAGES-II basin attributes and shapefiles (Falcone, 2011)
