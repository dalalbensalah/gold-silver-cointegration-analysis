import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen

from stationarity import adf_test


def engle_granger_test(y: pd.Series, x: pd.Series) -> pd.DataFrame:
    """
    Engle-Granger cointegration test (statsmodels implementation).

    H0: no cointegration between y and x. The test implicitly
    estimates the long-run regression, then applies an ADF test
    (with adjusted critical values) to the residual.

    Parameters
    ----------
    y, x : pd.Series
        Level series (e.g. log_gold, log_silver), same index.

    Returns
    -------
    pd.DataFrame
        One row summarizing statistic, p-value, critical thresholds
        and conclusion.
    """

    data = pd.concat([y, x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()

    score, pvalue, crit = coint(data.iloc[:, 0], data.iloc[:, 1])

    result = pd.DataFrame([{
        "Method": "Engle-Granger",
        "Null hypothesis": "No cointegration",
        "Statistic": round(score, 4),
        "p-value": round(pvalue, 4),
        "Critical threshold 1%": round(crit[0], 4),
        "Critical threshold 5%": round(crit[1], 4),
        "Critical threshold 10%": round(crit[2], 4),
        "Decision": "Reject H0" if pvalue < 0.05 else "Fail to reject H0",
        "Conclusion": "Cointegration" if pvalue < 0.05 else "No cointegration",
    }])

    return result


def ols_cointegration_residual(
    y: pd.Series,
    x: pd.Series,
) -> tuple[float, float, pd.Series, dict]:
    """
    Manual OLS cointegration regression and ADF test on the residual.

    Alternative, more pedagogical approach to engle_granger_test: the
    long-run relationship y = alpha + beta * x is estimated by simple
    OLS, then the residual is tested for stationarity (ADF with
    constant). Serves as a cross-check against statsmodels' `coint`
    implementation, which does not use exactly the same procedure
    (MacKinnon critical values specific to cointegration).

    Parameters
    ----------
    y, x : pd.Series
        Level series, same index (e.g. log_gold explained by
        log_silver, as in the notebook).

    Returns
    -------
    tuple
        (alpha_hat, beta_hat, residual, ADF result on the residual)
    """

    data = pd.concat([y, x], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    y_, x_ = data.iloc[:, 0], data.iloc[:, 1]

    beta_hat = ((x_ - x_.mean()) * (y_ - y_.mean())).sum() / ((x_ - x_.mean()) ** 2).sum()
    alpha_hat = y_.mean() - beta_hat * x_.mean()
    residual = y_ - alpha_hat - beta_hat * x_

    adf_res = adf_test(residual, regression="c")

    return alpha_hat, beta_hat, residual, adf_res


def johansen_all_specs(
    data: pd.DataFrame | dict[str, pd.DataFrame],
    variables: list[str] = ["log_gold", "log_silver"],
    det_orders: tuple[int, ...] = (0, 1),
    k_ar_diffs: tuple[int, ...] = (0, 1, 2),
    signif_index: int = 1,
) -> pd.DataFrame:
    """
    Johansen cointegration test (trace statistic) across several
    specifications, for one or more sub-samples.

    For each (det_order, k_ar_diff) combination, the cointegration
    rank is determined by sequentially comparing the trace statistic
    to the critical values, starting from r=0.

    Parameters
    ----------
    data : pd.DataFrame or dict[str, pd.DataFrame]
        A single sample, or a dictionary {period_name: DataFrame}
        to test several sub-periods in one pass.
    variables : list[str]
        Columns of the system (e.g. log-prices).
    det_orders : tuple[int, ...]
        Deterministic terms to test (0 = no trend in the cointegrated
        data, 1 = linear trend).
    k_ar_diffs : tuple[int, ...]
        Numbers of lags in differences to test.
    signif_index : int
        Critical threshold column in coint_johansen.cvt:
        0 = 10%, 1 = 5%, 2 = 1%.

    Returns
    -------
    pd.DataFrame
        One row per (period, det_order, k_ar_diff), with trace
        statistics, thresholds, decisions and retained rank.
    """

    results = []

    data_dict = data if isinstance(data, dict) else {"Global": data}

    for label, df_raw in data_dict.items():
        df = df_raw[variables].dropna()

        for det_order in det_orders:
            for k_ar_diff in k_ar_diffs:
                joh = coint_johansen(df, det_order=det_order, k_ar_diff=k_ar_diff)

                rank = 0
                for i in range(len(joh.lr1)):
                    if joh.lr1[i] > joh.cvt[i, signif_index]:
                        rank = i + 1
                    else:
                        break

                if rank == 0:
                    conclusion = "No cointegration"
                elif rank == 1:
                    conclusion = "One cointegration relationship"
                else:
                    conclusion = f"{rank} cointegration relationships"

                results.append({
                    "Period": label,
                    "det_order": det_order,
                    "k_ar_diff": k_ar_diff,
                    "Trace r<=0": round(joh.lr1[0], 4),
                    "Crit 5% r<=0": round(joh.cvt[0, signif_index], 4),
                    "Decision r<=0": (
                        "Reject H0" if joh.lr1[0] > joh.cvt[0, signif_index]
                        else "Fail to reject H0"
                    ),
                    "Trace r<=1": round(joh.lr1[1], 4),
                    "Crit 5% r<=1": round(joh.cvt[1, signif_index], 4),
                    "Decision r<=1": (
                        "Reject H0" if joh.lr1[1] > joh.cvt[1, signif_index]
                        else "Fail to reject H0"
                    ),
                    "Retained rank": rank,
                    "Conclusion": conclusion,
                })

    return pd.DataFrame(results)


def johansen_monthly_robustness(
    periods: dict[str, pd.DataFrame],
    variables: list[str] = ["log_gold", "log_silver"],
    det_orders: tuple[int, ...] = (0, 1),
    k_ar_diffs: tuple[int, ...] = (0, 1, 2),
    signif_index: int = 1,
    min_obs: int = 36,
) -> pd.DataFrame:
    """
    Johansen test replicated on monthly data, by sub-period, to
    assess the robustness of results to the observation frequency.

    Sub-periods with fewer than min_obs observations are skipped
    (test not reliable on too few points) and flagged in the
    "Remark" column rather than silently dropped.

    Parameters
    ----------
    periods : dict[str, pd.DataFrame]
        {period_name: monthly DataFrame}.
    variables : list[str]
        Columns of the system.
    det_orders, k_ar_diffs : tuple[int, ...]
        Specifications to test (see johansen_all_specs).
    signif_index : int
        0 = 10%, 1 = 5%, 2 = 1%.
    min_obs : int
        Minimum number of monthly observations required to run the
        test on a sub-period.

    Returns
    -------
    pd.DataFrame
        One row per (period, det_order, k_ar_diff), with the
        retained cointegration rank or a remark in case of an
        insufficient sample / error.
    """

    results = []
    signif_label = {0: "10%", 1: "5%", 2: "1%"}[signif_index]

    for period_name, df_period in periods.items():
        data = df_period[variables].replace([np.inf, -np.inf], np.nan).dropna()
        n_obs = len(data)

        for det_order in det_orders:
            for k_ar_diff in k_ar_diffs:
                if n_obs < min_obs:
                    results.append({
                        "Period": period_name,
                        "Obs": n_obs,
                        "det_order": det_order,
                        "k_ar_diff": k_ar_diff,
                        "Trace r=0": np.nan,
                        f"Crit {signif_label} r=0": np.nan,
                        "Cointegration": np.nan,
                        "Retained rank": np.nan,
                        "Remark": "Insufficient monthly sample to run the test",
                    })
                    continue

                try:
                    joh = coint_johansen(data, det_order=det_order, k_ar_diff=k_ar_diff)

                    rank = 0
                    for i in range(len(joh.lr1)):
                        if joh.lr1[i] > joh.cvt[i, signif_index]:
                            rank = i + 1
                        else:
                            break

                    results.append({
                        "Period": period_name,
                        "Obs": n_obs,
                        "det_order": det_order,
                        "k_ar_diff": k_ar_diff,
                        "Trace r=0": round(joh.lr1[0], 4),
                        f"Crit {signif_label} r=0": round(joh.cvt[0, signif_index], 4),
                        "Cointegration": rank >= 1,
                        "Retained rank": rank,
                        "Remark": "Sufficient monthly sample",
                    })

                except Exception as e:
                    results.append({
                        "Period": period_name,
                        "Obs": n_obs,
                        "det_order": det_order,
                        "k_ar_diff": k_ar_diff,
                        "Trace r=0": np.nan,
                        f"Crit {signif_label} r=0": np.nan,
                        "Cointegration": np.nan,
                        "Retained rank": np.nan,
                        "Remark": f"Error: {e}",
                    })

    return pd.DataFrame(results)


def rolling_cointegration(
    data: pd.DataFrame,
    target: str,
    features: list[str],
    window: int = 1000,
) -> pd.DataFrame:
    """
    Engle-Granger cointegration test applied over rolling windows,
    generalized to 2 or more variables, to study the temporal
    stability of cointegration.

    At each date t (starting from t = window), the test is run on
    the `window` preceding observations. Useful for detecting local
    cointegration episodes masked by a non-significant global test
    over the full period.

    Both cases below are indeed the Engle-Granger method (long-run
    regression + unit root test on the residual), which naturally
    generalizes to several explanatory variables (Engle & Granger,
    1987). The difference lies in the critical values used at the
    residual-testing step — this is not an implementation detail, it
    changes the statistical guarantees of the test:

    - `len(features) == 1` (bivariate case): uses statsmodels'
      `coint()`, which applies MacKinnon (1996) critical values,
      specifically calibrated for a cointegration regression
      residual.
    - `len(features) >= 2` (multivariate case): `coint()` does not
      support more than two series; we therefore fall back to
      `adfuller()` on the OLS regression residual, with generic ADF
      critical values, not adjusted for the number of regressors.
      This tends to bias the test toward too easily rejecting H0
      (more permissive cointegration detection than a correct
      adjustment would give).

    The "Method" column of the result indicates which of the two
    variants was used, so that this distinction remains visible
    downstream (tables, plots) without having to read the source
    code.

    Parameters
    ----------
    data : pd.DataFrame
        Level data, indexed by date, containing target and features.
    target : str
        Explained variable (e.g. "log_gold").
    features : list[str]
        Explanatory variable(s) (e.g. ["log_silver"] for the
        bivariate case, ["log_silver", "log_dxy", "us10y"] for the
        multivariate case).
    window : int
        Rolling window size (number of observations).

    Returns
    -------
    pd.DataFrame
        Indexed by window end date, with p-value, method used,
        decision and conclusion for each window.
    """

    cols = [target] + features
    data_clean = data[cols].replace([np.inf, -np.inf], np.nan).dropna()

    method = (
        "Engle-Granger (MacKinnon crit. values)"
        if len(features) == 1
        else "Engle-Granger multivariate (generic ADF crit. values)"
    )

    rows = []

    for i in range(window, len(data_clean)):
        sample = data_clean.iloc[i - window:i]

        if len(features) == 1:
            _, pvalue, _ = coint(sample[target], sample[features[0]])
        else:
            X = sm.add_constant(sample[features])
            y = sample[target]
            residual = sm.OLS(y, X).fit().resid
            _, pvalue, *_ = adfuller(residual)

        rows.append({
            "Date": data_clean.index[i],
            "p_value": pvalue,
            "Method": method,
            "Decision": "Reject H0" if pvalue < 0.05 else "Fail to reject H0",
            "Conclusion": "Cointegration" if pvalue < 0.05 else "No cointegration",
        })

    return pd.DataFrame(rows).set_index("Date")