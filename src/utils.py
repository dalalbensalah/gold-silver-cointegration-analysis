import numpy as np
import pandas as pd
from scipy.stats import chi2
import matplotlib.pyplot as plt


def level_metrics(pred: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """
    Computes RMSE, MAE and MAPE (%) between predictions and actual values, column by column.

    Used to evaluate the out-of-sample performance of VAR/VECM models
    after converting forecasts back to price levels
    (see notebooks 03 and 04, validation and test).

    Parameters
    ----------
    pred : pd.DataFrame
        Predicted values (same columns and index as `actual`).
    actual : pd.DataFrame
        Actually observed values.

    Returns
    -------
    pd.DataFrame
        Indexed by variable, columns RMSE / MAE / MAPE_%.
    """

    err = pred - actual
    rows = []

    for col in pred.columns:
        rows.append({
            "Variable": col,
            "RMSE": np.sqrt((err[col] ** 2).mean()),
            "MAE": err[col].abs().mean(),
            "MAPE_%": (err[col].abs() / actual[col].abs()).mean() * 100,
        })

    return pd.DataFrame(rows).set_index("Variable")


def find_significant_intervals(
    p_values: pd.Series,
    threshold: float = 0.05,
    min_days: int = 0,
) -> pd.DataFrame:
    """
    Identifies contiguous time intervals where a series of p-values
    stays below a given threshold (e.g. cointegration episodes detected
    by a rolling test).

    Reproduces the episode-detection logic used for the bivariate
    rolling Engle-Granger test (see notebook 02) and the multivariate
    one (see notebook 03), where the same code block was duplicated
    identically.

    Parameters
    ----------
    p_values : pd.Series
        Series indexed by date, p-value values (e.g. output of
        rolling_cointegration()["p_value"]).
    threshold : float
        Significance threshold (0.05 by default).
    min_days : int
        Minimum duration (in calendar days) for an interval to be
        kept. 0 = no filter.

    Returns
    -------
    pd.DataFrame
        Columns: Start, End, Duration_days, N_observations.
        Sorted by descending duration.
    """

    significant = p_values < threshold

    intervals = []
    start = None

    for date, is_sig in significant.items():
        if is_sig and start is None:
            start = date
        elif (not is_sig) and (start is not None):
            end = date
            intervals.append((start, end))
            start = None

    if start is not None:
        intervals.append((start, significant.index[-1]))

    rows = []
    for s, e in intervals:
        duration = (e - s).days
        if duration < min_days:
            continue
        n_obs = p_values.loc[s:e].shape[0]
        rows.append({
            "Start": s,
            "End": e,
            "Duration_days": duration,
            "N_observations": n_obs,
        })

    intervals_df = pd.DataFrame(rows, columns=["Start", "End", "Duration_days", "N_observations"])

    if not intervals_df.empty:
        intervals_df = intervals_df.sort_values("Duration_days", ascending=False).reset_index(drop=True)

    return intervals_df


def lr_test_var_lags(data: pd.DataFrame, max_p: int = 5) -> pd.DataFrame:
    """
    Likelihood ratio (LR) tests between nested VAR(p-1) vs VAR(p) models.

    Reproduces the logic duplicated between the bivariate framework
    (see notebook 03, section 7.3) and the multivariate one (see
    section 5), which compares VAR models of increasing dimension to
    determine whether adding a lag significantly improves the fit.

    Parameters
    ----------
    data : pd.DataFrame
        Stationary data (first differences), no NaN/inf.
    max_p : int
        Maximum number of lags tested.

    Returns
    -------
    pd.DataFrame
        One row per VAR(p-1) vs VAR(p) comparison, with LR statistic,
        degrees of freedom, p-value and decision at the 5% level.
    """

    from statsmodels.tsa.api import VAR

    k = data.shape[1]
    df_lr = k * k

    models = {p: VAR(data).fit(p) for p in range(1, max_p + 1)}

    results = []

    for p in range(2, max_p + 1):
        ll_p = models[p].llf
        ll_p_1 = models[p - 1].llf

        lr_stat = 2 * (ll_p - ll_p_1)
        p_value = 1 - chi2.cdf(lr_stat, df_lr)

        results.append({
            "Test": f"VAR({p-1}) vs VAR({p})",
            "H0": f"VAR({p-1}) is sufficient",
            "H1": f"VAR({p}) improves the model",
            "LR_stat": round(lr_stat, 4),
            "df": df_lr,
            "p_value": round(p_value, 4),
            "Decision (5%)": "Reject H0" if p_value < 0.05 else "Fail to reject H0",
        })

    return pd.DataFrame(results)

def save_current_fig(name: str, folder: str = "../figures", dpi: int = 150) -> None:
    """
    Save the currently active matplotlib figure to the project's figures/
    folder, in PNG format.

    Parameters
    ----------
    name : str
        File name (without extension), e.g. "rolling_cointegration".
    folder : str
        Path to the destination folder (defaults to ../figures from
        a notebook).
    dpi : int
        Export resolution.
    """
    import os
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f"{folder}/{name}.png", dpi=dpi, bbox_inches="tight")