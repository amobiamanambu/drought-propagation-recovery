%% event_level_stepwise_combined.m
% =========================================================================
% EVENT-LEVEL STEPWISE REGRESSION: DYNAMIC + STATIC PREDICTORS
% =========================================================================
% Each row = one matched drought event (N ≈ 75,640)
% Dynamic predictors vary per event; static predictors repeat per basin.
%
% Predictors: 4 event-level dynamic + ~26 GAGES-II static attributes
% Targets:    propagation_lag_months, recovery_lag_months
% Subsets:    All / Ref / Non-ref / Ecoregion × Class
% Timescales: 3-month and 6-month
%
% Requires in workspace:
%   matched_pairs_dynamic   (75,640 × N event-level table)
% =========================================================================

clearvars -except matched_pairs_dynamic
clc;

fprintf('##################################################################\n');
fprintf('  EVENT-LEVEL STEPWISE: DYNAMIC + STATIC PREDICTORS\n');
fprintf('##################################################################\n');

%% ==================== 1. READ GAGES-II STATIC ATTRIBUTES ================
fprintf('\n1. Reading GAGES-II attribute files...\n');

csv_dir = 'C:\Users\patan\OneDrive\Documents\MATLAB\gridmet\gridmet_shapefile\gagesii_csv';

file_vars = { ...
    'conterm_hydro.txt',        {'STAID','STREAMS_KM_SQ_KM','STRAHLER_MAX','BFI_AVE','CONTACT','PERDUN','PERHOR','TOPWET','RUNAVE7100'}; ...
    'conterm_climate.txt',      {'STAID','PPTAVG_BASIN','T_AVG_BASIN','RH_BASIN','PET','SNOW_PCT_PRECIP','PRECIP_SEAS_IND'}; ...
    'conterm_topo.txt',         {'STAID','ELEV_MEAN_M_BASIN','DRAIN_SQKM','SLOPE_PCT','RRMEAN','ASPECT_EASTNESS','ASPECT_NORTHNESS'}; ...
    'conterm_bas_classif.txt',  {'STAID','GEOL_REEDBUSH_DOM_PCT','HGA','HGB','HGC','HGD','PERMAVE','RFACT','BDAVE','AWCAVE','ROCKDEPAVE','WTDEPAVE'}; ...
    'conterm_bas_morph.txt',    {'STAID','BAS_COMPACTNESS'}; ...
    'conterm_lc06_basin.txt',   {'STAID','FORESTNLCD06','CROPSNLCD06','GRASSNLCD06','SHRUBNLCD06','WOODYWETNLCD06','EMERGWETNLCD06','DEVNLCD06'}; ...
    'conterm_hydromod_dams.txt',{'STAID','NDAMS_2009','STOR_NID_2009','RAW_DIS_NEAREST_DAM','RAW_AVG_DIS_ALLDAMS','RAW_DIS_NEAREST_MAJ_DAM'} ...
};

static_attrs = table();
for f = 1:size(file_vars,1)
    fname = fullfile(csv_dir, file_vars{f,1});
    cols = file_vars{f,2};
    
    fopts = detectImportOptions(fname);
    fopts.VariableNamingRule = 'preserve';
    fopts = setvartype(fopts, 'STAID', 'string');
    tmp = readtable(fname, fopts);
    
    avail_cols = intersect(cols, tmp.Properties.VariableNames);
    tmp = tmp(:, avail_cols);
    tmp.STAID = compose("%08s", tmp.STAID);
    
    if isempty(static_attrs)
        static_attrs = tmp;
    else
        static_attrs = outerjoin(static_attrs, tmp, 'Keys', 'STAID', 'MergeKeys', true);
    end
end

static_attrs.Properties.VariableNames{'STAID'} = 'GAGE_ID';
fprintf('   Loaded %d basins x %d static attributes\n', height(static_attrs), width(static_attrs)-1);

%% ==================== 2. LOAD CLASS AND ECOREGION =======================
fprintf('\n2. Loading CLASS and ecoregion...\n');

classif = readtable(fullfile(csv_dir, 'conterm_bas_classif.txt'));
classif.GAGE_ID = compose("%08s", string(classif.STAID));
classif.CLASS = string(classif.CLASS);
classif = classif(:, {'GAGE_ID', 'CLASS'});

shp = shaperead('C:\Users\patan\OneDrive\Documents\MATLAB\gridmet\gridmet_shapefile\gagesII_9322_point_shapefile\gagesII_9322_sept30_2011.shp');
shp_tbl = struct2table(shp);
shp_tbl.GAGE_ID = compose("%08s", string(shp_tbl.STAID));
eco = shp_tbl(:, {'GAGE_ID', 'AGGECOREGI'});
eco.Properties.VariableNames{'AGGECOREGI'} = 'ecoregion';
eco.ecoregion = string(eco.ecoregion);

fprintf('   CLASS: %d basins, Ecoregion: %d basins\n', height(classif), height(eco));

%% ==================== 3. PREPARE EVENT-LEVEL TABLE ======================
fprintf('\n3. Merging static attributes into event-level table...\n');

MPD = matched_pairs_dynamic;

% Standardize GAGE_ID to zero-padded 8-char strings
if ~isstring(MPD.GAGE_ID)
    MPD.GAGE_ID = string(MPD.GAGE_ID);
end
MPD.GAGE_ID = compose("%08s", MPD.GAGE_ID);

fprintf('   Events before merge: %d\n', height(MPD));

% Merge static attributes (left join to keep all events)
E = outerjoin(MPD, static_attrs, 'Keys', 'GAGE_ID', 'MergeKeys', true, 'Type', 'left');

% Add CLASS if not already present
if ~ismember('CLASS', E.Properties.VariableNames)
    E = outerjoin(E, classif, 'Keys', 'GAGE_ID', 'MergeKeys', true, 'Type', 'left');
end

% Add ecoregion if not already present
if ~ismember('ecoregion', E.Properties.VariableNames)
    E = outerjoin(E, eco, 'Keys', 'GAGE_ID', 'MergeKeys', true, 'Type', 'left');
end

fprintf('   Events after merge: %d x %d columns\n', height(E), width(E));

% Verify CLASS/ecoregion
if ismember('CLASS', E.Properties.VariableNames)
    E.CLASS = string(E.CLASS);
    fprintf('   Ref events: %d  |  Non-ref events: %d\n', ...
        sum(E.CLASS == "Ref"), sum(E.CLASS == "Non-ref"));
end
if ismember('ecoregion', E.Properties.VariableNames)
    E.ecoregion = string(E.ecoregion);
end

%% ==================== 4. DEFINE ANALYSIS PARAMETERS =====================

% Dynamic predictors (event-level, vary per event)
dyn_vars = {'dyn_BFI', 'dyn_recession_k_days', 'dyn_cv_Q', 'antecedent_ssi'};

% Human modification variables to exclude from Ref models
human_vars = {'NDAMS_2009','STOR_NID_2009','RAW_DIS_NEAREST_DAM', ...
              'RAW_AVG_DIS_ALLDAMS','RAW_DIS_NEAREST_MAJ_DAM','DEVNLCD06'};

% Columns to exclude from predictor pool (identifiers, targets, dates, etc.)
base_exclude = {'GAGE_ID','timescale','CLASS','ecoregion', ...
                'spei_onset','spei_termination','spei_duration', ...
                'ssi_onset','ssi_termination','ssi_duration', ...
                'propagation_lag_months','recovery_lag_months', ...
                'match_type','spei_severity','ssi_severity', ...
                'spei_rebound_3mo','spei_rebound_6mo', ...
                'dyn_window_days','dyn_zero_flow_fraction', ...
                'BFI_AVE'};  % exclude static BFI (collinear with dyn_BFI)

% Targets
targets = struct();
targets(1).name = 'Propagation Lag';
targets(1).dep_var = 'propagation_lag_months';
targets(1).extra_exclude = {'recovery_lag_months'};

targets(2).name = 'Recovery Lag';
targets(2).dep_var = 'recovery_lag_months';
targets(2).extra_exclude = {'propagation_lag_months'};

%% ==================== 5. LOOP OVER TIMESCALES AND TARGETS ===============
for ts = [3, 6]
    fprintf('\n##########################################################\n');
    fprintf('TIMESCALE: %d-month\n', ts);
    fprintf('##########################################################\n');
    
    % Subset to this timescale
    Et = E(E.timescale == ts, :);
    
    ref_mask = Et.CLASS == "Ref";
    nr_mask = Et.CLASS == "Non-ref";
    
    fprintf('   Events: %d  (Ref: %d  |  Non-ref: %d)\n', ...
        height(Et), sum(ref_mask), sum(nr_mask));
    
    for t = 1:length(targets)
        fprintf('\n==========================================================\n');
        fprintf('TARGET: %s - %d-month\n', targets(t).name, ts);
        fprintf('==========================================================\n');
        
        dep_var = targets(t).dep_var;
        
        % Build exclusion list
        all_targets = {'propagation_lag_months', 'recovery_lag_months'};
        this_exclude = unique([base_exclude, targets(t).extra_exclude, all_targets]);
        
        % Get available predictor names
        all_pred = setdiff(Et.Properties.VariableNames, this_exclude);
        
        % Screen predictors on full pool
        [screened_preds, ~] = screen_predictors(Et, all_pred);
        
        % Ref pool: remove human modification variables
        screened_preds_ref = setdiff(screened_preds, human_vars, 'stable');
        
        % Count dynamic vs static
        dyn_flags = ismember(screened_preds, dyn_vars);
        fprintf('   Predictor pool (All/Non-ref): %d variables\n', length(screened_preds));
        fprintf('   Predictor pool (Ref):         %d variables\n', length(screened_preds_ref));
        fprintf('   Dynamic (%d): %s\n', sum(dyn_flags), strjoin(screened_preds(dyn_flags), ', '));
        fprintf('   Static  (%d): %s\n', sum(~dyn_flags), strjoin(screened_preds(~dyn_flags), ', '));
        
        % ---- ALL EVENTS ----
        fprintf('\n  ------ ALL / REF / NON-REF ------\n');
        run_stepwise_event(Et, screened_preds, dep_var, 'All', dyn_vars);
        
        % ---- REFERENCE ----
        run_stepwise_event(Et(ref_mask,:), screened_preds_ref, dep_var, 'Ref', dyn_vars);
        
        % ---- NON-REFERENCE ----
        run_stepwise_event(Et(nr_mask,:), screened_preds, dep_var, 'Non-ref', dyn_vars);
        
        % ---- ECOREGION x REFERENCE ----
        fprintf('\n  ------ ECOREGION x REFERENCE ------\n');
        ecos = unique(Et.ecoregion(ref_mask & Et.ecoregion ~= "" & ~ismissing(Et.ecoregion)));
        for e = 1:length(ecos)
            sub = Et(ref_mask & Et.ecoregion == ecos(e), :);
            run_stepwise_event(sub, screened_preds_ref, dep_var, char(ecos(e)), dyn_vars);
        end
        
        % ---- ECOREGION x NON-REFERENCE ----
        fprintf('\n  ------ ECOREGION x NON-REFERENCE ------\n');
        ecos_nr = unique(Et.ecoregion(nr_mask & Et.ecoregion ~= "" & ~ismissing(Et.ecoregion)));
        for e = 1:length(ecos_nr)
            sub = Et(nr_mask & Et.ecoregion == ecos_nr(e), :);
            run_stepwise_event(sub, screened_preds, dep_var, char(ecos_nr(e)), dyn_vars);
        end
    end
end

fprintf('\n##########################################################\n');
fprintf('  EVENT-LEVEL ANALYSIS COMPLETE\n');
fprintf('##########################################################\n');


%% ==================== HELPER FUNCTIONS ==================================

function [screened_names, X_out] = screen_predictors(M, pred_names)
    % Screen predictor pool: remove high-missing, zero-variance,
    % collinear (|r|>0.85), and high-VIF (>10) predictors
    
    avail = intersect(pred_names, M.Properties.VariableNames, 'stable');
    X = M{:, avail};
    
    % Remove >20% missing or zero variance
    miss_frac = mean(isnan(X));
    zero_var = std(X, 'omitnan') < 1e-6;
    keep = miss_frac < 0.2 & ~zero_var;
    avail = avail(keep);
    X = X(:, keep);
    
    % Correlation screening (|r| > 0.85) — use subsample for speed
    n = size(X, 1);
    if n > 5000
        rng(42);
        idx = randsample(n, 5000);
        Xsub = X(idx, :);
    else
        Xsub = X;
    end
    R = corrcoef(Xsub, 'Rows', 'pairwise');
    drop = false(1, length(avail));
    for i = 1:size(R,1)
        for j = i+1:size(R,2)
            if abs(R(i,j)) > 0.85 && ~drop(j)
                drop(j) = true;
            end
        end
    end
    avail = avail(~drop);
    X = X(:, ~drop);
    
    % VIF screening (>10) — use subsample for speed
    for iter = 1:10
        if n > 5000
            Xsub = X(idx, :);
        else
            Xsub = X;
        end
        cc = all(~isnan(Xsub), 2);
        Xcc = Xsub(cc,:);
        np = size(Xcc, 2);
        if np < 3; break; end
        vif = zeros(1, np);
        for p = 1:np
            others = setdiff(1:np, p);
            mdl = fitlm(Xcc(:,others), Xcc(:,p));
            vif(p) = 1 / (1 - mdl.Rsquared.Ordinary);
        end
        [mv, worst] = max(vif);
        if mv > 10
            avail(worst) = [];
            X(:, worst) = [];
        else
            break;
        end
    end
    
    screened_names = avail;
    X_out = X;
end


function run_stepwise_event(sub, pred_names, dep_var, label, dyn_vars)
    avail = intersect(pred_names, sub.Properties.VariableNames, 'stable');
    X = sub{:, avail};
    y = sub.(dep_var);
    
    cc = all(~isnan(X), 2) & ~isnan(y);
    X = X(cc,:);
    y = y(cc);
    N = length(y);
    
    fprintf('\n    %-15s (N = %d events)\n', label, N);
    
    if N < 50
        fprintf('    *** Too few events, skipping ***\n');
        return;
    end
    
    % Remove zero-variance in this subset
    pstd = std(X);
    keep = pstd > 1e-6;
    X = X(:, keep);
    names_use = avail(keep);
    
    if isempty(names_use)
        fprintf('    *** No valid predictors ***\n');
        return;
    end
    
    % For large N, use subsample for stepwise fitting, then refit on full
    if N > 10000
        rng(42);
        fit_idx = randsample(N, 10000);
        X_fit = X(fit_idx, :);
        y_fit = y(fit_idx);
        fprintf('    (stepwise selection on 10,000 subsample, coefficients on full N)\n');
    else
        X_fit = X;
        y_fit = y;
    end
    
    [~, ~, ~, inmodel, ~, ~, ~] = ...
        stepwisefit(X_fit, y_fit, 'penter', 0.05, 'premove', 0.10, 'display', 'off');
    
    sel_idx = find(inmodel);
    n_sel = length(sel_idx);
    
    if n_sel == 0
        fprintf('    *** No predictors entered ***\n');
        return;
    end
    
    % Refit selected predictors on full data using fitlm for proper stats
    sel_names = names_use(sel_idx);
    X_sel = X(:, sel_idx);
    
    tbl_fit = array2table(X_sel, 'VariableNames', sel_names);
    tbl_fit.y = y;
    
    mdl = fitlm(tbl_fit, 'ResponseVar', 'y');
    
    R2 = mdl.Rsquared.Ordinary;
    adjR2 = mdl.Rsquared.Adjusted;
    rmse_val = mdl.RMSE;
    
    fprintf('    R² = %.4f  |  Adj R² = %.4f  |  RMSE = %.4f\n', R2, adjR2, rmse_val);
    
    % Standardized betas
    X_std = (X_sel - mean(X_sel)) ./ std(X_sel);
    y_std = (y - mean(y)) / std(y);
    beta_std = X_std \ y_std;
    
    % Get p-values from fitlm (skip intercept at row 1)
    coef_tbl = mdl.Coefficients;
    p_vals = coef_tbl.pValue(2:end);  % skip intercept
    coeffs = coef_tbl.Estimate(2:end);
    
    [~, imp_order] = sort(abs(beta_std), 'descend');
    
    fprintf('    %-25s %8s %10s %8s  %s\n', 'Variable', 'Beta', 'Coeff', 'p-value', 'Type');
    fprintf('    %s\n', repmat('-', 1, 65));
    for k = imp_order(:)'
        vname = sel_names{k};
        if ismember(vname, dyn_vars)
            vtype = 'DYN';
        else
            vtype = 'STATIC';
        end
        fprintf('    %-25s %8.4f %10.6f %8.4f  %s\n', ...
            vname, beta_std(k), coeffs(k), p_vals(k), vtype);
    end
end