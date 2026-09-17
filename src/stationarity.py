import warnings

import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss


def adf_test(series: pd.Series, regression: str = "c") -> dict:
    """
    Augmented Dickey-Fuller (ADF) test.

    H0: presence of a unit root (non-stationary series).
    The number of lags is automatically selected by AIC.

    Parameters
    ----------
    series : pd.Series
        Series to test.
    regression : str
        Deterministic term included in the test regression:
        "n" (none), "c" (constant), "ct" (constant + trend).
        The notebook uses "n" for log-price levels (no drift
        assumed a priori) and "c" for cointegration residuals
        (possible non-zero mean).

    Returns
    -------
    dict
        ADF statistic, p-value, number of lags used,
        5% critical value.
    """

    stat, pvalue, usedlag, nobs, crit, icbest = adfuller(
        series.dropna(), regression=regression, autolag="AIC"
    )

    return {
        "ADF stat": stat,
        "p-value": pvalue,
        "lags": usedlag,
        "crit_5%": crit["5%"],
    }


def kpss_test(series: pd.Series, regression: str = "c") -> dict:
    """
    KPSS test (Kwiatkowski-Phillips-Schmidt-Shin).

    H0: the series is stationary (test complementary to ADF,
    reversed hypotheses). The number of lags is automatically
    selected ("auto").

    Parameters
    ----------
    series : pd.Series
        Series to test.
    regression : str
        "c" (stationarity around a constant) or "ct"
        (stationarity around a deterministic trend).

    Returns
    -------
    dict
        KPSS statistic, p-value, number of lags, 5% critical value.
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InterpolationWarning)
        stat, pvalue, lags, crit = kpss(
            series.dropna(), regression=regression, nlags="auto"
        )

    return {
        "KPSS stat": stat,
        "p-value KPSS": pvalue,
        "lags": lags,
        "crit_5%": crit["5%"],
    }


def stationarity_table(
    data: pd.DataFrame,
    adf_regression: str = "n",
    kpss_regression: str = "c",
) -> pd.DataFrame:
    """
    Applies ADF and KPSS to each column of a DataFrame and summarizes
    the conclusions.

    The two tests are complementary: ADF tests H0 = unit root,
    KPSS tests H0 = stationarity. A series is considered robustly
    stationary if both tests agree (ADF rejection AND KPSS
    non-rejection).

    Parameters
    ----------
    data : pd.DataFrame
        Columns to test (e.g. log-prices in level or in differences).
    adf_regression : str
        Deterministic term for ADF (see adf_test). Default "n"
        because used in level, without constant, in the notebook
        (see section 5, stationarity_table).
    kpss_regression : str
        Deterministic term for KPSS (see kpss_test).

    Returns
    -------
    pd.DataFrame
        One row per variable, indexed by column name, with
        statistics, p-values and conclusions of both tests.
    """

    rows = []

    for col in data.columns:
        adf_res = adf_test(data[col], regression=adf_regression)
        kpss_res = kpss_test(data[col], regression=kpss_regression)

        rows.append({
            "Series": col,
            "ADF stat": adf_res["ADF stat"],
            "ADF p-value": adf_res["p-value"],
            "ADF conclusion": (
                "Stationary" if adf_res["p-value"] < 0.05
                else "Non-stationary"
            ),
            "KPSS stat": kpss_res["KPSS stat"],
            "KPSS p-value": kpss_res["p-value KPSS"],
            "KPSS conclusion": (
                "Non-stationary" if kpss_res["p-value KPSS"] < 0.05
                else "Stationary"
            ),
        })

    return pd.DataFrame(rows).set_index("Series")