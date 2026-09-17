import matplotlib.pyplot as plt
import pandas as pd


### Data exploration

def plot_series_panel(
    data: pd.DataFrame,
    columns: list[str],
    titles: list[str] | None = None,
    ylabels: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
) -> None:
    """
    Plots several time series stacked vertically, one panel per
    column, with a shared time axis.

    Generalizes the multi-asset plots from the exploration notebook
    (e.g. gold/silver in Part I, gold/silver/dxy/us10y in Part II).

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame indexed by date, containing the columns to plot.
    columns : list[str]
        Columns to plot, one per panel, in order.
    titles : list[str], optional
        Titles for each panel. Defaults to the column name.
    ylabels : list[str], optional
        Y-axis labels for each panel. Defaults to the column name.
    figsize : tuple[int, int], optional
        Figure size. Defaults to (14, 3 * number of columns).

    Returns
    -------
    None
        Displays the figure (plt.show()).
    """
    n = len(columns)
    titles = titles or columns
    ylabels = ylabels or columns
    figsize = figsize or (14, 3 * n)

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]

    for ax, col, title, ylabel in zip(axes, columns, titles, ylabels):
        ax.plot(data.index, data[col])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_log_prices_and_ratio(df: pd.DataFrame, price_cols: list[str] = ["log_gold", "log_silver"], ratio_col: str = "log_ratio") -> None:
    """
    Plots the overlaid log-prices, then the log-ratio separately.

    Reproduces the plots from section 3.1 of the bivariate
    exploration notebook: direct comparison of the two log-prices,
    then their relative gap (log-ratio) read on a second plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by date.
    price_cols : list[str]
        Log-price columns to overlay on the first plot.
    ratio_col : str
        Log-ratio column, plotted alone on the second plot.

    Returns
    -------
    None
        Displays both figures (plt.show()).
    """
    plt.figure(figsize=(12, 5))
    for col in price_cols:
        plt.plot(df.index, df[col], label=col)
    plt.title("Log-prices")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df[ratio_col], color="steelblue")
    plt.title("Log-ratio")
    plt.grid(alpha=0.3)
    plt.show()


def plot_imputation_comparison(
    df_ffill: pd.DataFrame,
    df_interpolate: pd.DataFrame,
    column: str,
    zoom_window: tuple[str, str] | None = None,
) -> None:
    """
    Visually compares two imputation methods (ffill vs interpolation)
    on a given column, with an optional zoom on a window with a high
    density of holidays.

    Used to justify the imputation method chosen in the study
    (see preprocessing.impute_missing).

    Parameters
    ----------
    df_ffill, df_interpolate : pd.DataFrame
        DataFrames imputed respectively by ffill and by interpolation
        (same columns, same index).
    column : str
        Column to compare.
    zoom_window : tuple[str, str], optional
        (start_date, end_date) for a second, zoomed-in plot on a
        narrower window (e.g. around year-end). If None, only the
        full-period plot is displayed.

    Returns
    -------
    None
        Displays one or two figures (plt.show()).
    """
    def _plot(data_ffill, data_interp, title):
        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        axes[0].plot(data_ffill, label="ffill", linewidth=2, color="red", marker="o")
        axes[0].set_title(f"{title} — Forward fill")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].plot(data_interp, label="interpolation", linewidth=2, color="steelblue", marker="o")
        axes[1].set_title(f"{title} — Linear interpolation")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    _plot(df_ffill[column], df_interpolate[column], f"{column} — full period")

    if zoom_window is not None:
        start, end = zoom_window
        _plot(df_ffill[column][start:end], df_interpolate[column][start:end], f"{column} — zoom {start} to {end}")


### Cointegration

def plot_rolling_cointegration(
    rolling_result: pd.DataFrame,
    intervals: pd.DataFrame | None = None,
    threshold: float = 0.05,
) -> None:
    """
    Plots the rolling Engle-Granger test's p-value, with the
    significance threshold and detected cointegration intervals
    highlighted.

    Parameters
    ----------
    rolling_result : pd.DataFrame
        Output of cointegration.rolling_engle_granger (must contain
        the "p-value" column, indexed by date).
    intervals : pd.DataFrame, optional
        Output of cointegration.detect_cointegration_intervals
        (columns "Start", "End"). If provided, the intervals are
        highlighted.
    threshold : float
        Significance threshold displayed as a dashed line.

    Returns
    -------
    None
        Displays the figure (plt.show()).
    """
    plt.figure(figsize=(14, 6))

    plt.plot(rolling_result.index, rolling_result["p-value"], label="Rolling Engle-Granger p-value")
    plt.axhline(threshold, linestyle="--", color="red", label=f"Threshold {threshold:.0%}")

    if intervals is not None:
        for _, row in intervals.iterrows():
            plt.axvspan(row["Start"], row["End"], alpha=0.15, color="steelblue")

    plt.title("Rolling Engle-Granger Cointegration Test")
    plt.ylabel("p-value")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


### VAR / VECM — forecasting

def plot_forecast_levels(
    act_levels: pd.DataFrame,
    pred_levels: pd.DataFrame,
    columns: list[str],
    model_label: str,
    sample_label: str,
    variable_labels: dict[str, str] | None = None,
) -> None:
    """
    Plots, for each variable, actual vs predicted levels over a given
    period (validation or test).

    Generic function that replaces the 4 nearly identical code blocks
    in the notebook (validation/test x VAR/VECM): a single
    maintenance point for the forecast plots' formatting.

    Parameters
    ----------
    act_levels, pred_levels : pd.DataFrame
        Actual and predicted levels, same columns and index (output
        of var_vecm.recursive_one_step_forecast_var/_vecm).
    columns : list[str]
        Columns to plot, one per panel.
    model_label : str
        Model name shown in the legend (e.g. "VAR(1)", "VECM").
    sample_label : str
        Sample name shown in the title (e.g. "Validation", "Test").
    variable_labels : dict[str, str], optional
        Display labels per column (e.g. {"log_gold": "Gold"}).
        Defaults to the raw column name.

    Returns
    -------
    None
        Displays the figure (plt.show()).
    """
    variable_labels = variable_labels or {col: col for col in columns}

    fig, axes = plt.subplots(len(columns), 1, figsize=(12, 4 * len(columns)), sharex=True)
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(act_levels.index, act_levels[col], label="Actual", linewidth=1.8, color="black")
        ax.plot(pred_levels.index, pred_levels[col], label=f"Forecast {model_label}",
                linewidth=1.6, linestyle="--", color="steelblue")
        ax.set_title(f"{sample_label} — {variable_labels[col]}")
        ax.legend(loc="upper left")
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


### VAR / VECM — impulse response functions

def plot_irf(IRF, variable_names: list[str], H: int, model_label: str = "") -> None:
    """
    Plots the orthogonalized impulse response functions: one figure
    per structural shock, with all system variables overlaid.

    Generic function that replaces plot_irf_vecm and the manual VAR
    IRF plotting (cells 91 and 121 of the notebook) — same logic,
    whether the IRF comes from a manual computation
    (var_vecm.compute_irf) or from vecm_res.irf().orth_irfs.

    Parameters
    ----------
    IRF : np.ndarray
        Array of shape (H+1, k, k): IRF[h, i, j] = response of
        variable i to shock j, at horizon h.
    variable_names : list[str]
        Names of the system's variables, in the order of IRF's axes.
    H : int
        Horizon displayed (must be <= IRF.shape[0] - 1).
    model_label : str, optional
        Detail shown in the title (e.g. "VAR(1)", "VECM").

    Returns
    -------
    None
        Displays one figure per structural shock (plt.show()).
    """
    suffix = f" ({model_label})" if model_label else ""

    for j, shock in enumerate(variable_names):
        plt.figure(figsize=(8, 4))

        for i, var in enumerate(variable_names):
            plt.plot(range(H + 1), IRF[:H + 1, i, j], label=var, linewidth=1.7)

        plt.axhline(0, color="black", linewidth=1, linestyle="--")
        plt.title(f"Responses to an orthogonal shock on {shock}{suffix}")
        plt.xlabel("Horizon (business days)")
        plt.ylabel("Response")
        plt.legend()
        plt.grid(alpha=0.2)
        plt.tight_layout()
        plt.show()