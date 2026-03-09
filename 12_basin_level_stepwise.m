%% dynamic_analysis_full_combined.m
%% THIS IS BASIN LEVEL ANALYSIS; USING STATIC VARIABLES, AND DYNAMIC VARIABLES AVERAGED BY BASIN
% =========================================================================
% COMBINED DYNAMIC + STATIC STEPWISE REGRESSION
% =========================================================================
% Reproduces the stepwise_all_four_phases.m analysis but using the
% event-level matched_pairs_dynamic table as the source for dynamic vars.
%
% Predictors: 4 aggregated dynamic vars + ~28 GAGES-II static attributes
% Targets: buffered_pct, independent_pct, propagation_lag, recovery_lag
% Subsets: All / Ref / Non-ref / Ecoregion × Class
% Timescales: 3-month and 6-month
%
% Requires in workspace:
%   matched_pairs_dynamic        (event-level table with dynamic vars)
%   drought_basin_classification (basin-level with buffered/indep %)
% =========================================================================

clearvars -except matched_pairs_dynamic drought_basin_classification
clc;

fprintf('##################################################################\n');
fprintf('  COMBINED DYNAMIC + STATIC STEPWISE REGRESSION\n');
fprintf('##################################################################\n');

%% ==================== 1. AGGREGATE DYNAMIC VARS TO BASIN LEVEL ==========
fprintf('\n1. Aggregating dynamic variables to basin level (mean)...\n');

MPD = matched_pairs_dynamic;

dyn_agg_vars = {'dyn_BFI','dyn_recession_k_days','dyn_cv_Q', ...
                'antecedent_ssi','propagation_lag_months','recovery_lag_months'};

agg_names = {'dyn_BFI','dyn_recession_k','dyn_cv_Q', ...
             'mean_antecedent_ssi','mean_propagation_lag','mean_recovery_lag'};

dyn_basin = table();
for ts = [3, 6]
    sub = MPD(MPD.timescale == ts, :);
    gages = unique(sub.GAGE_ID);
    n = length(gages);
    
    tbl = table();
    tbl.GAGE_ID = gages;
    tbl.timescale = repmat(ts, n, 1);
    
    for v = 1:length(dyn_agg_vars)
        vals = nan(n, 1);
        for g = 1:n
            mask = sub.GAGE_ID == gages(g);
            x = sub.(dyn_agg_vars{v})(mask);
            vals(g) = mean(x, 'omitnan');
        end
        tbl.(agg_names{v}) = vals;
    end
    
    dyn_basin = [dyn_basin; tbl]; %#ok<AGROW>
end

% Zero-pad GAGE_IDs to 8 characters (leading zeros were stripped earlier)
dyn_basin.GAGE_ID = compose("%08s", dyn_basin.GAGE_ID);

fprintf('   Aggregated %d basin-timescale records\n', height(dyn_basin));

%% ==================== 2. READ GAGES-II ATTRIBUTES =======================
fprintf('\n2. Reading GAGES-II attribute files...\n');

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

%% ==================== 3. LOAD CLASS AND ECOREGION =======================
fprintf('\n3. Loading CLASS and ecoregion...\n');

% CLASS
classif = readtable(fullfile(csv_dir, 'conterm_bas_classif.txt'));
classif.GAGE_ID = compose("%08s", string(classif.STAID));
classif.CLASS = string(classif.CLASS);
classif = classif(:, {'GAGE_ID', 'CLASS'});

% Ecoregion from shapefile
shp = shaperead('C:\Users\patan\OneDrive\Documents\MATLAB\gridmet\gridmet_shapefile\gagesII_9322_point_shapefile\gagesII_9322_sept30_2011.shp');
shp_tbl = struct2table(shp);
shp_tbl.GAGE_ID = compose("%08s", string(shp_tbl.STAID));
eco = shp_tbl(:, {'GAGE_ID', 'AGGECOREGI'});
eco.Properties.VariableNames{'AGGECOREGI'} = 'ecoregion';
eco.ecoregion = string(eco.ecoregion);

fprintf('   CLASS: %d basins, Ecoregion: %d basins\n', height(classif), height(eco));

%% ==================== 4. PREPARE BASIN-LEVEL TARGETS ====================
fprintf('\n4. Preparing basin-level targets from drought_basin_classification...\n');

DBC = drought_basin_classification;
if ~isstring(DBC.GAGE_ID)
    DBC.GAGE_ID = string(DBC.GAGE_ID);
end
DBC.GAGE_ID = compose("%08s", regexprep(DBC.GAGE_ID, '^0+', ''));

% Build results table with targets per timescale
results_all = table();
for ts = [3, 6]
    ts_str = num2str(ts);
    tbl = table();
    tbl.GAGE_ID = DBC.GAGE_ID;
    tbl.timescale = repmat(ts, height(DBC), 1);
    tbl.buffered_pct = DBC.(sprintf('buffered_%s_pct', ts_str));
    tbl.independent_pct = DBC.(sprintf('independent_%s_pct', ts_str));
    results_all = [results_all; tbl]; %#ok<AGROW>
end

fprintf('   %d basin-timescale records\n', height(results_all));

%% ==================== 5. HUMAN MODIFICATION VARIABLES ===================
human_vars = {'NDAMS_2009','STOR_NID_2009','RAW_DIS_NEAREST_DAM', ...
              'RAW_AVG_DIS_ALLDAMS','RAW_DIS_NEAREST_MAJ_DAM','DEVNLCD06'};

%% ==================== 6. LOOP OVER TIMESCALES ===========================
for ts = [3, 6]
    fprintf('\n##########################################################\n');
    fprintf('TIMESCALE: %d-month\n', ts);
    fprintf('##########################################################\n');
    
    % Results for this timescale
    res_ts = results_all(results_all.timescale == ts, :);
    
    % Dynamic aggregates for this timescale
    dyn_ts = dyn_basin(dyn_basin.timescale == ts, :);
    dyn_ts.timescale = [];
    
    % Merge: results + dynamic + static + CLASS + ecoregion
    M = innerjoin(res_ts, dyn_ts, 'Keys', 'GAGE_ID');
    M = innerjoin(M, static_attrs, 'Keys', 'GAGE_ID');
    M = outerjoin(M, classif, 'Keys', 'GAGE_ID', 'MergeKeys', true, 'Type', 'left');
    M = outerjoin(M, eco, 'Keys', 'GAGE_ID', 'MergeKeys', true, 'Type', 'left');
    
    fprintf('   Merged: %d basins\n', height(M));
    
    % Identify ref/non-ref masks
    ref_mask = M.CLASS == "Ref";
    nr_mask = M.CLASS == "Non-ref";
    
    fprintf('   Ref: %d  |  Non-ref: %d\n', sum(ref_mask), sum(nr_mask));
    
    %% Define targets
    base_exclude = {'GAGE_ID','timescale','CLASS','ecoregion', ...
                    'buffered_pct','independent_pct','BFI_AVE', ...
                    'mean_recovery_lag','mean_propagation_lag'};
    
    targets = struct();
    
    targets(1).name = 'Buffered %';
    targets(1).phase = 'Phase 1 - Gate';
    targets(1).dep_var = 'buffered_pct';
    targets(1).extra_exclude = {'mean_recovery_lag','mean_propagation_lag'};
    
    targets(2).name = 'Independent %';
    targets(2).phase = 'Independent Initiation';
    targets(2).dep_var = 'independent_pct';
    targets(2).extra_exclude = {'mean_recovery_lag','mean_propagation_lag'};
    
    targets(3).name = 'Propagation Lag';
    targets(3).phase = 'Phase 2 - Pathway';
    targets(3).dep_var = 'mean_propagation_lag';
    targets(3).extra_exclude = {'mean_recovery_lag','buffered_pct','independent_pct'};
    
    targets(4).name = 'Recovery Lag';
    targets(4).phase = 'Phase 3 - Brake';
    targets(4).dep_var = 'mean_recovery_lag';
    targets(4).extra_exclude = {'mean_propagation_lag','buffered_pct','independent_pct'};
    
    for t = 1:length(targets)
        fprintf('\n==========================================================\n');
        fprintf('TARGET: %s (%s) - %d-month\n', targets(t).name, targets(t).phase, ts);
        fprintf('==========================================================\n');
        
        dep_var = targets(t).dep_var;
        
        all_targets = {'buffered_pct','independent_pct','mean_recovery_lag','mean_propagation_lag'};
        this_exclude = unique([base_exclude, targets(t).extra_exclude, all_targets]);
        
        all_pred = setdiff(M.Properties.VariableNames, this_exclude);
        
        % Screen predictors (full pool)
        [screened_preds, ~] = screen_predictors(M, all_pred);
        
        % Ref pool: remove human modification variables
        screened_preds_ref = setdiff(screened_preds, human_vars, 'stable');
        
        fprintf('   Predictor pool (All/Non-ref): %d variables\n', length(screened_preds));
        fprintf('   Predictor pool (Ref):         %d variables (human vars removed)\n', length(screened_preds_ref));
        dyn_flags = startsWith(screened_preds, 'dyn_') | startsWith(screened_preds, 'mean_');
        fprintf('   Dynamic (%d): %s\n', sum(dyn_flags), strjoin(screened_preds(dyn_flags), ', '));
        static_flags = ~dyn_flags;
        fprintf('   Static  (%d): %s\n', sum(static_flags), strjoin(screened_preds(static_flags), ', '));
        
        % ---- ALL BASINS ----
        fprintf('\n  ------ ALL / REF / NON-REF ------\n');
        run_stepwise_report(M, screened_preds, dep_var, 'All');
        
        % ---- REFERENCE ----
        run_stepwise_report(M(ref_mask,:), screened_preds_ref, dep_var, 'Ref');
        
        % ---- NON-REFERENCE ----
        run_stepwise_report(M(nr_mask,:), screened_preds, dep_var, 'Non-ref');
        
        % ---- ECOREGION: REFERENCE ----
        fprintf('\n  ------ ECOREGION x REFERENCE ------\n');
        ecos = unique(M.ecoregion(ref_mask & M.ecoregion ~= "" & ~ismissing(M.ecoregion)));
        for e = 1:length(ecos)
            eco_mask = ref_mask & M.ecoregion == ecos(e);
            sub = M(eco_mask,:);
            run_stepwise_report(sub, screened_preds_ref, dep_var, char(ecos(e)));
        end
        
        % ---- ECOREGION: NON-REFERENCE ----
        fprintf('\n  ------ ECOREGION x NON-REFERENCE ------\n');
        ecos_nr = unique(M.ecoregion(nr_mask & M.ecoregion ~= "" & ~ismissing(M.ecoregion)));
        for e = 1:length(ecos_nr)
            eco_mask = nr_mask & M.ecoregion == ecos_nr(e);
            sub = M(eco_mask,:);
            run_stepwise_report(sub, screened_preds, dep_var, char(ecos_nr(e)));
        end
    end
end

fprintf('\n##########################################################\n');
fprintf('  ANALYSIS COMPLETE\n');
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
    
    % Correlation screening (|r| > 0.85)
    R = corrcoef(X, 'Rows', 'pairwise');
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
    
    % VIF screening (>10)
    for iter = 1:10
        cc = all(~isnan(X), 2);
        Xcc = X(cc,:);
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


function run_stepwise_report(sub, pred_names, dep_var, label)
    avail = intersect(pred_names, sub.Properties.VariableNames, 'stable');
    X = sub{:, avail};
    y = sub.(dep_var);
    
    cc = all(~isnan(X), 2) & ~isnan(y);
    X = X(cc,:);
    y = y(cc);
    N = length(y);
    
    fprintf('\n    %-15s (N = %d)\n', label, N);
    
    if N < 30
        fprintf('    *** Too few basins, skipping ***\n');
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
    
    [b, se, pval, inmodel, stats, ~, ~] = ...
        stepwisefit(X, y, 'penter', 0.05, 'premove', 0.10, 'display', 'off');
    
    sel_idx = find(inmodel);
    n_sel = length(sel_idx);
    
    R2 = 1 - stats.SSresid / stats.SStotal;
    if N > n_sel + 1
        adjR2 = 1 - (stats.SSresid/(N - n_sel - 1)) / (stats.SStotal/(N - 1));
    else
        adjR2 = NaN;
    end
    
    fprintf('    R² = %.4f  |  Adj R² = %.4f  |  RMSE = %.4f\n', R2, adjR2, stats.rmse);
    
    if n_sel == 0
        fprintf('    *** No predictors entered ***\n');
        return;
    end
    
    sel_names = names_use(sel_idx);
    sel_b = b(sel_idx);
    sel_p = pval(sel_idx);
    
    % Standardized betas
    X_sel = X(:, sel_idx);
    X_std = (X_sel - mean(X_sel)) ./ std(X_sel);
    y_std = (y - mean(y)) / std(y);
    beta_std = X_std \ y_std;
    
    [~, imp_order] = sort(abs(beta_std), 'descend');
    
    fprintf('    %-25s %8s %10s %8s  %s\n', 'Variable', 'Beta', 'Coeff', 'p-value', 'Type');
    fprintf('    %s\n', repmat('-', 1, 65));
    for k = imp_order(:)'
        vname = sel_names{k};
        if startsWith(vname, 'dyn_') || startsWith(vname, 'mean_')
            vtype = 'DYN';
        else
            vtype = 'STATIC';
        end
        fprintf('    %-25s %8.4f %10.4f %8.4f  %s\n', ...
            vname, beta_std(k), sel_b(k), sel_p(k), vtype);
    end
end