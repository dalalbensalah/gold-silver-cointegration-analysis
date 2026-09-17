import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import VECM, select_order, coint_johansen



# VAR : sélection de retards et diagnostics


def select_var_lag_order(data: pd.DataFrame, maxlags: int = 10):
    """
    Select the VAR lag order using information criteria
    (AIC, BIC, HQIC, FPE).

    Parameters
    ----------
    data : pd.DataFrame
        Stationary data (e.g., dlog_gold, dlog_silver returns).
    maxlags : int
        Maximum number of lags tested.

    Returns
    -------
    statsmodels LagOrderResults
        Object containing the lags recommended by each criterion
        (accessible via .selected_orders, printable via .summary()).
    """
    
    return VAR(data).select_order(maxlags=maxlags)


def var_diagnostics_table(data: pd.DataFrame, candidates: list[int], nlags_whiteness: int = 12) -> pd.DataFrame:
    """
    Fit a VAR for each candidate lag order and summarize residual
    diagnostics (whiteness, normality) and information criteria.

    Parameters
    ----------
    data : pd.DataFrame
        Stationary data used for estimation.
    candidates : list[int]
        Values of p to compare (e.g., [1, 2, 3, 4, 5]).
    nlags_whiteness : int
        Number of lags tested for residual autocorrelation
        (whiteness test, Portmanteau).

    Returns
    -------
    pd.DataFrame
        One row per value of p: AIC, BIC, whiteness test p-value,
        and multivariate residual normality test p-value.
    """
    rows = []

    for p in candidates:
        res = VAR(data).fit(p)
        white = res.test_whiteness(nlags=nlags_whiteness)
        normal = res.test_normality()

        rows.append({
            "Lag": p,
            "AIC": res.aic,
            "BIC": res.bic,
            "Whiteness p-value": white.pvalue,
            "Normalité p-value": normal.pvalue,
        })

    return pd.DataFrame(rows).set_index("Lag")


def lr_test_nested_var(data: pd.DataFrame, max_p: int = 5) -> pd.DataFrame:
    """
    Likelihood ratio (LR) tests between nested VAR(p) and VAR(p-1) models.

    H0: the additional coefficients in VAR(p) relative to VAR(p-1)
    are zero (the extra lag does not significantly improve the fit).

    Parameters
    ----------
    data : pd.DataFrame
        Stationary data.
    max_p : int
        Largest lag order tested (the tests compare
        p=2 vs p=1, p=3 vs p=2, ..., up to max_p).

    Returns
    -------
    pd.DataFrame
        LR statistic, degrees of freedom, p-value, and decision for
        each comparison (p vs p-1).
    """
    k = data.shape[1]
    df_lr = k * k

    models = {p: VAR(data).fit(p) for p in range(1, max_p + 1)}

    rows = []
    for p in range(2, max_p + 1):
        ll_p = models[p].llf
        ll_p_1 = models[p - 1].llf

        lr_stat = 2 * (ll_p - ll_p_1)
        p_value = 1 - chi2.cdf(lr_stat, df_lr)

        rows.append({
            "Comparaison": f"VAR({p}) vs VAR({p - 1})",
            "LR stat": lr_stat,
            "df": df_lr,
            "p-value": p_value,
            "Décision": "Reject H0 (additional lag is significant)" if p_value < 0.05
                        else "Fail to reject H0",
        })

    return pd.DataFrame(rows)



# VAR : prévision récursive one-step (walk-forward)


def recursive_one_step_forecast_var(
    diff_data_full: pd.DataFrame,
    log_data_full: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    p: int,
) -> dict:
    """
    Recursive one-step-ahead forecast of a VAR estimated on differences,
    with log-level reconstruction by adding the predicted return to the
    last ACTUALLY OBSERVED level (t-1).

    At each date t, the VAR is re-estimated on the available history
    (up to t-1 only, no future information leakage), then used to predict
    the return at t. The predicted log level is reconstructed by adding
    this predicted return to the actual observed level at t-1 — not to
    the predicted level from the previous step, which would cause error
    accumulation not representative of a genuine one-step-ahead forecast.

    Parameters
    ----------
    diff_data_full : pd.DataFrame
        Returns (log-price differences) over the full train+val+test
        period, same index as log_data_full.
    log_data_full : pd.DataFrame
        Log-price levels, same index as diff_data_full.
    start_idx, end_idx : int
        Integer position bounds of the period to forecast.
    p : int
        Number of VAR lags.

    Returns
    -------
    dict
        {"pred_diff", "act_diff"}: DataFrames of predicted/actual returns.
        {"pred_levels", "act_levels"}: DataFrames of reconstructed levels.
    """
    cols = diff_data_full.columns.tolist()

    pred_diff_rows, act_diff_rows, dates = [], [], []

    for t in range(start_idx, end_idx):
        history = diff_data_full.iloc[:t]
        model = VAR(history).fit(p)

        forecast = model.forecast(history.values[-p:], steps=1)[0]

        pred_diff_rows.append(forecast)
        act_diff_rows.append(diff_data_full.iloc[t].values)
        dates.append(diff_data_full.index[t])

    pred_diff = pd.DataFrame(pred_diff_rows, index=dates, columns=cols)
    act_diff = pd.DataFrame(act_diff_rows, index=dates, columns=cols)

    # Level reconstruction: OBSERVED level at t-1 + predicted return at t.
    # Each forecast is re-anchored on the actual data (not on the previous
    # forecast), consistent with a one-step-ahead logic.
    log_cols = log_data_full.columns.tolist()

    pred_levels_rows, act_levels_rows = [], []

    for date in dates:
        prev_level = log_data_full.loc[:date].iloc[-2].values  # actual level at t-1
        pred_levels_rows.append(prev_level + pred_diff.loc[date].values)
        act_levels_rows.append(log_data_full.loc[date].values)

    pred_levels = pd.DataFrame(pred_levels_rows, index=dates, columns=log_cols)
    act_levels = pd.DataFrame(act_levels_rows, index=dates, columns=log_cols)

    return {
        "pred_diff": pred_diff,
        "act_diff": act_diff,
        "pred_levels": pred_levels,
        "act_levels": act_levels,
    }


def forecast_accuracy_table(pred: pd.DataFrame, act: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize forecast accuracy per variable: RMSE, MAE, MAPE.

    MAPE is used as the main comparison metric across specifications
    since it is expressed in percentage terms, making it insensitive to
    scale differences between variables (e.g., gold vs. silver price).

    Parameters
    ----------
    pred, act : pd.DataFrame
        Predicted and actual values, same columns and index.

    Returns
    -------
    pd.DataFrame
        One row per variable: RMSE, MAE, MAPE (%).
    """
    rows = []
    for col in act.columns:
        err = act[col] - pred[col]
        rmse = np.sqrt((err ** 2).mean())
        mae = err.abs().mean()
        mape = (err.abs() / act[col].abs()).mean() * 100

        rows.append({"Variable": col, "RMSE": rmse, "MAE": mae, "MAPE_%": mape})

    return pd.DataFrame(rows).set_index("Variable")


# VAR — causality

def granger_causality_table(var_res, variables: list[str]) -> pd.DataFrame:
    """
    Granger causality tests for all ordered pairs of variables in the
    system.

    H0: past values of `causing` do not improve the forecast of `caused`
    (no Granger causality).

    Parameters
    ----------
    var_res : statsmodels VARResults
        Estimated VAR model (e.g., final_var_full).
    variables : list[str]
        Names of the system's variables.

    Returns
    -------
    pd.DataFrame
        One row per ordered pair (causing -> caused), with F statistic,
        p-value, and decision.
    """
    rows = []

    for caused in variables:
        for causing in variables:
            if caused == causing:
                continue

            test = var_res.test_causality(caused, [causing], kind="f")

            rows.append({
                "Causing": causing,
                "Caused": caused,
                "F stat": test.test_statistic,
                "p-value": test.pvalue,
                "Decision": "Reject H0 (causality)" if test.pvalue < 0.05 else "Fail to reject H0",
            })

    return pd.DataFrame(rows)


def instantaneous_causality_table(var_res, causing: list[str]) -> pd.DataFrame:
    """
    Instantaneous (symmetric) causality test between `causing` and the
    other variables of the system.

    H0: no contemporaneous correlation between the innovations of the
    tested variables (instantaneous independence of shocks).

    Parameters
    ----------
    var_res : statsmodels VARResults
        Estimated VAR model.
    causing : list[str]
        Subset of variables tested for instantaneous causality.

    Returns
    -------
    pd.DataFrame
        One row summarizing statistic, p-value, and decision.
    """
    test = var_res.test_inst_causality(causing=causing)

    return pd.DataFrame([{
        "Variables tested": " ↔ ".join(var_res.names),
        "Null hypothesis": "No instantaneous causality",
        "Statistic": test.test_statistic,
        "p-value": test.pvalue,
        "Decision": "Reject H0 (contemporaneous dependence)" if test.pvalue < 0.05 else "Fail to reject H0",
    }])


# VAR — orthogonal shocks, IRF and FEVD (manual computation)

def cholesky_decomposition(var_res) -> np.ndarray:
    """
    Cholesky decomposition of the residual covariance matrix, used to
    orthogonalize structural shocks.

    The order of the variables in var_res determines the imposed
    contemporaneous causal ordering (the first variable reacts only to
    its own shock; subsequent variables also react to the shocks of the
    preceding ones).

    Parameters
    ----------
    var_res : statsmodels VARResults
        Estimated VAR model.

    Returns
    -------
    np.ndarray
        Lower-triangular matrix P such that Sigma_u = P @ P.T
        (manual 2x2 implementation; for k > 2, prefer
        np.linalg.cholesky(Sigma_u)).
    """
    sigma_u = var_res.sigma_u.to_numpy()

    if sigma_u.shape == (2, 2):
        s11, s12, s22 = sigma_u[0, 0], sigma_u[0, 1], sigma_u[1, 1]
        p11 = np.sqrt(s11)
        p21 = s12 / p11
        p22 = np.sqrt(s22 - p21 ** 2)
        return np.array([[p11, 0.0], [p21, p22]])

    return np.linalg.cholesky(sigma_u)


def orthogonal_shocks(var_res, P: np.ndarray) -> pd.DataFrame:
    """
    Computes the orthogonal structural shocks from the VAR residuals and
    the Cholesky matrix.

    v_t = P^-1 @ u_t, where u_t are the VAR residuals. By construction,
    Var(v_t) should be close to the identity matrix (orthogonal, unit-
    variance shocks) — serves as a check on the decomposition.

    Parameters
    ----------
    var_res : statsmodels VARResults
        Estimated VAR model.
    P : np.ndarray
        Cholesky matrix (see cholesky_decomposition).

    Returns
    -------
    pd.DataFrame
        Orthogonal shocks, one column per structural variable
        (v1, v2, ...), same index as the VAR residuals.
    """
    u_t = var_res.resid.to_numpy()
    p_inv = np.linalg.inv(P)
    v_t = (p_inv @ u_t.T).T

    cols = [f"v{i + 1}" for i in range(v_t.shape[1])]
    return pd.DataFrame(v_t, index=var_res.resid.index, columns=cols)


def compute_irf(var_res, P: np.ndarray, H: int) -> np.ndarray:
    """
    Computes the orthogonalized impulse response functions of a VAR,
    from its moving-average (MA) representation.

    IRF[h, i, j] = response of variable i at horizon h following a unit
    structural shock to variable j at horizon 0.

    Parameters
    ----------
    var_res : statsmodels VARResults
        Estimated VAR model.
    P : np.ndarray
        Cholesky matrix (see cholesky_decomposition).
    H : int
        Maximum horizon (in periods).

    Returns
    -------
    np.ndarray
        Array of shape (H+1, k, k): orthogonalized IRFs.
    """
    k = var_res.neqs
    p = var_res.k_ar
    A = var_res.coefs

    Phi = np.zeros((H + 1, k, k))
    Phi[0] = np.eye(k)

    for h in range(1, H + 1):
        temp = np.zeros((k, k))
        for lag in range(1, min(h, p) + 1):
            temp += A[lag - 1] @ Phi[h - lag]
        Phi[h] = temp

    IRF = np.array([Phi[h] @ P for h in range(H + 1)])
    return IRF


def compute_fevd(IRF: np.ndarray, H: int) -> np.ndarray:
    """
    Forecast error variance decomposition (FEVD) from the orthogonalized
    IRFs.

    FEVD[h, i, j] = share of the forecast error variance of variable i
    at horizon h attributable to structural shock j.

    Parameters
    ----------
    IRF : np.ndarray
        Orthogonalized IRFs, shape (H+1, k, k) (see compute_irf).
    H : int
        Maximum horizon (must be <= IRF.shape[0] - 1).

    Returns
    -------
    np.ndarray
        Array of shape (H, k, k): FEVD, as proportions (sums to 1 over
        the shock axis, for each variable and horizon).
    """
    k = IRF.shape[1]
    FEVD = np.zeros((H, k, k))

    for h in range(1, H + 1):
        for i in range(k):
            denom = sum(np.sum(IRF[:h, i, m] ** 2) for m in range(k))
            for j in range(k):
                num = np.sum(IRF[:h, i, j] ** 2)
                FEVD[h - 1, i, j] = num / denom if denom != 0 else np.nan

    return FEVD


def fevd_table(FEVD: np.ndarray, variable_idx: int, variable_name: str, shock_names: list[str]) -> pd.DataFrame:
    """
    Formats the FEVD of a given variable as a readable table (in %, one
    row per horizon).

    Parameters
    ----------
    FEVD : np.ndarray
        Output of compute_fevd, shape (H, k, k).
    variable_idx : int
        Index of the explained variable (row in FEVD[:, i, :]).
    variable_name : str
        Variable name, for display/printing.
    shock_names : list[str]
        Names of the structural shocks (columns).

    Returns
    -------
    pd.DataFrame
        Indexed by horizon (1 to H), columns "Shock <name>", values in %.
    """
    H = FEVD.shape[0]
    table = pd.DataFrame(
        np.round(FEVD[:, variable_idx, :] * 100, 2),
        index=range(1, H + 1),
        columns=[f"Shock {name}" for name in shock_names],
    )
    table.index.name = f"Horizon (FEVD {variable_name})"
    return table


# VECM — episode identification and local estimation

def select_vecm_lag_order(data: pd.DataFrame, maxlags: int = 10, deterministic: str = "co"):
    """
    Selects the number of lagged differences (k_ar_diff) for a VECM.

    Parameters
    ----------
    data : pd.DataFrame
        Level series (log prices), over the chosen estimation window.
    maxlags : int
        Maximum number of lags tested.
    deterministic : str
        Deterministic term ("co" = constant restricted to the
        cointegration relation; statsmodels convention).

    Returns
    -------
    statsmodels LagOrderResults
        Object containing .aic, .bic, etc. (lag recommended by each
        criterion).
    """
    return select_order(data, maxlags=maxlags, deterministic=deterministic)


def fit_vecm(data: pd.DataFrame, k_ar_diff: int, coint_rank: int = 1, deterministic: str = "co"):
    """
    Estimates a VECM on a given data window.

    Parameters
    ----------
    data : pd.DataFrame
        Level series (log prices), over the estimation window.
    k_ar_diff : int
        Number of lagged differences.
    coint_rank : int
        Cointegration rank used (number of long-run relations).
    deterministic : str
        Deterministic term (see select_vecm_lag_order).

    Returns
    -------
    statsmodels VECMResults
        Estimated VECM model.
    """
    return VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic).fit()


def vecm_beta_alpha_tables(vecm_res, variable_names: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Formats the long-run (beta) and adjustment (alpha) coefficients of an
    estimated VECM, with standard errors, t-stats, and p-values.

    Parameters
    ----------
    vecm_res : statsmodels VECMResults
        Estimated VECM model (see fit_vecm).
    variable_names : list[str]
        Names of the system's variables, in the order used for
        estimation.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (beta_table, alpha_table) — cointegration vector and adjustment
        speeds, with associated inference.
    """
    beta_table = pd.DataFrame({
        "Variable": variable_names,
        "Beta": vecm_res.beta[:, 0],
        "Std_Error": vecm_res.stderr_beta[:, 0],
        "t_stat": vecm_res.tvalues_beta[:, 0],
        "p_value": vecm_res.pvalues_beta[:, 0],
    })

    alpha_table = pd.DataFrame({
        "Equation": [f"d{v}" for v in variable_names],
        "Alpha": vecm_res.alpha[:, 0],
        "Std_Error": vecm_res.stderr_alpha[:, 0],
        "t_stat": vecm_res.tvalues_alpha[:, 0],
        "p_value": vecm_res.pvalues_alpha[:, 0],
    })

    return beta_table, alpha_table


def normalized_long_run_relation(vecm_res, variable_names: list[str], normalize_on: int = 0) -> dict:
    """
    Normalizes the cointegration vector beta relative to a reference
    variable, for a direct reading of the long-run relation.

    The raw beta vector has no direct interpretation (it is defined up to
    a multiplicative constant). Normalizing on the variable at index
    `normalize_on` (0 = log_gold by default) allows the relation to be
    written as y = f(x), which is more readable for economic
    interpretation.

    Parameters
    ----------
    vecm_res : statsmodels VECMResults
        Estimated VECM model.
    variable_names : list[str]
        Names of the system's variables, in the order of vecm_res.beta.
    normalize_on : int
        Index of the variable to normalize on (0 by default, consistent
        with the log_gold-first convention used in the notebook).

    Returns
    -------
    dict
        {"beta_raw", "beta_normalized", "equation_str"}: raw vector,
        normalized vector, and the relation formatted as text.
    """
    beta_raw = vecm_res.beta[:, 0]
    beta_norm = beta_raw / beta_raw[normalize_on]

    ref_var = variable_names[normalize_on]
    other_terms = " + ".join(
        f"({beta_norm[i]:.4f}) * {variable_names[i]}"
        for i in range(len(variable_names)) if i != normalize_on
    )

    equation_str = f"{ref_var} = -({other_terms})"

    return {
        "beta_raw": beta_raw,
        "beta_normalized": beta_norm,
        "equation_str": equation_str,
    }


def compute_ect(data: pd.DataFrame, beta_normalized: np.ndarray, variable_names: list[str]) -> pd.Series:
    """
    Computes the error correction term (ECT) from the normalized
    cointegration vector.

    The ECT measures the instantaneous deviation from the estimated
    long-run relation: close to zero when the variables are at
    equilibrium, it moves away from zero during transitory imbalances,
    which the VECM corrects for via the alpha coefficients.

    Parameters
    ----------
    data : pd.DataFrame
        Level series used for estimation (same columns as
        variable_names).
    beta_normalized : np.ndarray
        Normalized cointegration vector (see
        normalized_long_run_relation).
    variable_names : list[str]
        Variable names, in the order of beta_normalized.

    Returns
    -------
    pd.Series
        ECT, same index as data.
    """
    ect = sum(beta_normalized[i] * data[variable_names[i]] for i in range(len(variable_names)))
    ect.name = "ECT"
    return ect


def vecm_summary_table(beta_table: pd.DataFrame, alpha_table: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenates the beta (long-run) and alpha (adjustment) tables into a
    single summary table, for display/export.

    Parameters
    ----------
    beta_table, alpha_table : pd.DataFrame
        Outputs of vecm_beta_alpha_tables.

    Returns
    -------
    pd.DataFrame
        Columns ["Block", "Name", "Coefficient", "Std_Error", "t_stat",
        "p_value"], one row per coefficient (beta then alpha).
    """
    beta_final = beta_table.rename(columns={"Variable": "Name", "Beta": "Coefficient"}).copy()
    beta_final["Block"] = "Long run (beta)"

    alpha_final = alpha_table.rename(columns={"Equation": "Name", "Alpha": "Coefficient"}).copy()
    alpha_final["Block"] = "Adjustment (alpha)"

    cols = ["Block", "Name", "Coefficient", "Std_Error", "t_stat", "p_value"]
    return pd.concat([beta_final[cols], alpha_final[cols]], axis=0).reset_index(drop=True)


def recursive_one_step_forecast_vecm(
    vecm_data_full: pd.DataFrame,
    log_cols: list[str],
    start_idx: int,
    end_idx: int,
    k_ar_diff: int,
    coint_rank: int,
    deterministic: str = "co",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Recursive one-step-ahead forecast of a VECM (walk-forward), analogous
    to recursive_one_step_forecast_var but for an error-correction model.

    At each date t, the VECM is re-estimated on the available history
    (up to t-1 only) and then used to predict the level at t. Expensive
    in computation time (a VECM is re-estimated at each step), so reserve
    for validation/test windows of reasonable size.

    Parameters
    ----------
    vecm_data_full : pd.DataFrame
        Level series over the full available period (must cover at least
        up to end_idx).
    log_cols : list[str]
        Columns of the system (e.g., ["log_gold", "log_silver"]).
    start_idx, end_idx : int
        Integer position bounds of the period to forecast.
    k_ar_diff : int
        Number of lagged differences in the VECM.
    coint_rank : int
        Cointegration rank used.
    deterministic : str
        Deterministic term (see select_vecm_lag_order).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (pred_log, act_log): predicted and realized log levels, indexed
        by date.
    """
    pred_log, act_log, idx = [], [], []

    for t in range(start_idx, end_idx):
        history = vecm_data_full[log_cols].iloc[:t]

        model = VECM(history, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=deterministic)
        res = model.fit()

        fc = res.predict(steps=1)
        pred_log.append(fc[0])
        act_log.append(vecm_data_full[log_cols].iloc[t].values)
        idx.append(vecm_data_full.index[t])

    pred_log_df = pd.DataFrame(pred_log, index=idx, columns=log_cols)
    act_log_df = pd.DataFrame(act_log, index=idx, columns=log_cols)

    return pred_log_df, act_log_df


def vecm_residual_diagnostics(vecm_res, variable_names: list[str], lags: int = 12) -> pd.DataFrame:
    """
    Ljung-Box test (residual whiteness) on each equation of an estimated
    VECM.

    H0: no residual autocorrelation up to lag `lags`. Complements the
    diagnostics performed on the VAR (test_whiteness) for the VECM case,
    whose residuals are not directly exposed via an equivalent method in
    statsmodels.

    Parameters
    ----------
    vecm_res : statsmodels VECMResults
        Estimated VECM model.
    variable_names : list[str]
        Names of the system's variables, in the order of the residual
        columns.
    lags : int
        Number of lags tested by the Ljung-Box test.

    Returns
    -------
    pd.DataFrame
        One row per variable: LB statistic, p-value, decision.
    """
    import statsmodels.api as sm

    resid = pd.DataFrame(vecm_res.resid, columns=variable_names)

    rows = []
    for col in variable_names:
        lb = sm.stats.acorr_ljungbox(resid[col], lags=[lags], return_df=True)
        stat = lb["lb_stat"].values[0]
        pvalue = lb["lb_pvalue"].values[0]

        rows.append({
            "Series": f"{col} (residuals)",
            "LB stat": stat,
            "p-value": pvalue,
            "Decision": "Reject H0 (autocorrelation)" if pvalue < 0.05 else "Fail to reject H0",
        })

    return pd.DataFrame(rows)

def var_irf_by_subperiod(periods, columns, p, H, min_obs_per_param=10):
    """
    Re-estimates a VAR(p) and its orthogonalized IRF separately on each
    sub-period, to check whether the impulse-response shape is stable
    across macro regimes or specific to the full-sample estimation.

    Periods with too few observations relative to the number of
    estimated parameters (k^2 * p) are skipped and reported, rather
    than silently dropped.

    Parameters
    ----------
    periods : dict[str, pd.DataFrame]
        {period_name: DataFrame}, each containing `columns` in the
        same transformation used for the retained VAR (differences).
    columns : list[str]
        Columns of the system, in the order used for the retained
        VAR (determines the Cholesky ordering).
    p : int
        Number of lags, held fixed across sub-periods for
        comparability with the full-sample specification.
    H : int
        Horizon for the IRF.
    min_obs_per_param : int
        Minimum number of observations per estimated parameter
        (k^2 * p) required to keep a sub-period.

    Returns
    -------
    dict[str, np.ndarray]
        {period_name: IRF array of shape (H+1, k, k)}, for periods
        with enough observations.
    """
    k = len(columns)
    n_params = k * k * p
    results = {}

    for name, df_period in periods.items():
        data = df_period[columns].replace([np.inf, -np.inf], np.nan).dropna()

        if len(data) < min_obs_per_param * n_params:
            print(f"Skipped '{name}': {len(data)} observations, "
                  f"below the {min_obs_per_param * n_params} threshold for VAR({p}).")
            continue

        res = VAR(data).fit(p)
        P = cholesky_decomposition(res)
        results[name] = compute_irf(res, P, H)

    return results


def irf_subperiod_comparison(irf_by_period, response_idx, shock_idx, response_name, shock_name):
    """
    Extracts one impulse-response path (response_idx to shock_idx) from
    each sub-period's IRF array, for side-by-side comparison.

    Parameters
    ----------
    irf_by_period : dict[str, np.ndarray]
        Output of var_irf_by_subperiod.
    response_idx, shock_idx : int
        Indices of the response and shock variables (order of
        `columns` passed to var_irf_by_subperiod).
    response_name, shock_name : str
        Names, for the output index label.

    Returns
    -------
    pd.DataFrame
        Indexed by horizon, one column per period: response of
        `response_name` to a shock on `shock_name`.
    """
    data = {period: IRF[:, response_idx, shock_idx] for period, IRF in irf_by_period.items()}
    table = pd.DataFrame(data)
    table.index.name = f"Horizon ({response_name} to {shock_name} shock)"
    return table