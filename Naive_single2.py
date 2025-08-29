# pip install pandas numpy matplotlib openpyxl

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG: point to your file/columns ----------------
EXCEL_PATH = Path("../Data/demand_data.xlsx")   # path to your Excel file
SHEET_NAME = "HISTORIC"                         # tab name
TIME_COL   = "DATE/HOUR"                        # timestamp column header
VALUE_COL  = "PRN_ABANCAY_138_VEG_IEOD"         # one series to forecast

# time/grid settings
FREQ = "30min"          # regular grid
PERIOD = 48             # 48 half-hours per day
FORECAST_DAYS = 2       # forecast horizon in days
K_DAYS_AVG = 0          # 0 = pure seasonal-naive; >0 = average last K days per slot (try 7)
# -------------------------------------------------------------------

# Quick peek to confirm columns
x = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME, nrows=1, engine="openpyxl")
print(x.columns.tolist())

def load_and_regularise(path: Path) -> pd.Series:
    """
    Robust loader:
      - Read two columns
      - Handle comma decimals
      - Parse dates
      - Resample to 30min grid
      - Interpolate gaps, then pad edges
    """
    df = pd.read_excel(
        path,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
        usecols=[TIME_COL, VALUE_COL],
        decimal=",",                       # safe even if decimals use dots
        dtype={VALUE_COL: "float64"},
    )

    print("[debug] columns:", list(df.columns))
    print("[debug] head:\n", df.head(3))

    # Parse timestamp robustly (infer_datetime_format no longer needed)
    t = pd.to_datetime(df[TIME_COL], errors="coerce", dayfirst=True)
    v = pd.to_numeric(df[VALUE_COL], errors="coerce")

    s = pd.Series(v.values, index=t, name=VALUE_COL).dropna().sort_index()

    # Aggregate duplicates if any
    if not s.index.is_unique:
        s = s.groupby(level=0).mean()

    if len(s) == 0:
        print("[warn] After parsing, series is empty. Check sheet/column names and date/decimal formats.")
        return s

    # Regularise to exact 30-min grid
    s30 = s.resample(FREQ).mean()

    # Interpolate + pad edges
    n_missing_before = int(s30.isna().sum())
    s30 = s30.interpolate(method="time").ffill().bfill().astype("float64")

    print(f"[regularise] points={len(s30)}, filled_NaNs={n_missing_before}")
    dt = s30.index.to_series().diff().dropna()
    if len(dt):
        print(f"[grid] min_step={dt.min()}, max_step={dt.max()} (should be {FREQ})")
    else:
        print("[grid] not enough data to compute steps")

    return s30

def seasonal_naive_forecast(series: pd.Series, steps: int, period: int, k_days_avg: int = 0) -> pd.Series:
    """
    If we have >= period points, do seasonal-naive (or K-day average).
    If not, fall back to last-value persistence.
    """
    last_stamp = series.index[-1]
    future_index = pd.date_range(start=last_stamp + pd.Timedelta(FREQ), periods=steps, freq=FREQ)

    if len(series) < period:
        print(f"[warn] Only {len(series)} points available (<{period}). Using last-value persistence.")
        return pd.Series(np.full(steps, series.iloc[-1], dtype=float), index=future_index, name="forecast")

    if k_days_avg <= 0:
        template = series[-period:].to_numpy()
    else:
        tail = series[-period * k_days_avg:]
        slots = np.arange(len(tail)) % period
        means = np.zeros(period, dtype=float)
        for slot in range(period):
            means[slot] = tail[slots == slot].mean()
        template = means

    reps = int(np.ceil(steps / period))
    tiled = np.tile(template, reps)[:steps]
    return pd.Series(tiled, index=future_index, name="forecast")

def plot_last_14_days_with_forecast(hist: pd.Series, fc: pd.Series, out_path: Path):
    lookback = 14 * PERIOD
    tail = hist.iloc[-lookback:] if len(hist) > lookback else hist

    plt.figure()
    tail.plot(label="history")
    fc.plot(label=f"seasonal-naive (+{len(fc)} steps)")
    plt.title(f"{VALUE_COL}: {FREQ} seasonal-naive forecast")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {out_path.resolve()}")

def run_diagnostics(s30: pd.Series, period: int = 48):
    """
    Missing-slot report + 7-day backtest.
    Uses shifts so prediction indexes align with actuals.
    """
    # ---- Missing slots BEFORE interpolation ----
    raw = pd.read_excel(
        EXCEL_PATH,
        sheet_name=SHEET_NAME,
        engine="openpyxl",
        usecols=[TIME_COL, VALUE_COL],
    )
    raw[TIME_COL] = pd.to_datetime(raw[TIME_COL], dayfirst=True, errors="coerce")
    raw = raw.dropna().set_index(TIME_COL)[VALUE_COL].sort_index()
    grid = pd.date_range(raw.index.min().floor(FREQ), raw.index.max().ceil(FREQ), freq=FREQ)
    missing = grid.difference(raw.index.unique())
    print(f"[diag] Missing half-hours before interpolation: {len(missing)}")
    if len(missing):
        miss = pd.Series(1, index=missing)
        blocks = (miss.index.to_series().diff() != pd.Timedelta(FREQ)).cumsum()
        report = miss.groupby(blocks).agg(start=lambda s: s.index.min(), end=lambda s: s.index.max(), slots="count")
        print("[diag] Largest missing blocks:\n", report.sort_values("slots", ascending=False).head(5))

    # ---- 7-day backtest on the fixed grid ----
    if len(s30) < 8 * period:
        print("[diag] Not enough history for a 7-day backtest; skipping.")
        return

    actual = s30.iloc[-7 * period:]          # last 7 days
    pred_sn = actual.shift(period)            # yesterday's same slot
    shifts = [actual.shift(p * period) for p in range(1, 8)]
    pred_k7 = pd.concat(shifts, axis=1).mean(axis=1)

    def mape(y, yhat):
        idx = y.index.intersection(yhat.index)
        y_, yhat_ = y.loc[idx], yhat.loc[idx]
        mask = (y_ != 0) & (~y_.isna()) & (~yhat_.isna())
        return float((np.abs((y_[mask] - yhat_[mask]) / y_[mask])).mean() * 100)

    def rmse(y, yhat):
        idx = y.index.intersection(yhat.index)
        y_, yhat_ = y.loc[idx], yhat.loc[idx]
        mask = (~y_.isna()) & (~yhat_.isna())
        return float(np.sqrt(((y_[mask] - yhat_[mask]) ** 2).mean()))

    print("[diag] Validation on last 7 days:")
    print("  seasonal-naive :", "MAPE", round(mape(actual, pred_sn), 2), "% | RMSE", round(rmse(actual, pred_sn), 3))
    print("  7-day per-slot :", "MAPE", round(mape(actual, pred_k7), 2), "% | RMSE", round(rmse(actual, pred_k7), 3))

def main():
    print(f"[load] {EXCEL_PATH.resolve()}")
    s30 = load_and_regularise(EXCEL_PATH)

    # Diagnostics (index-safe)
    run_diagnostics(s30, period=PERIOD)

    # Forecast
    steps = FORECAST_DAYS * PERIOD
    fc = seasonal_naive_forecast(s30, steps=steps, period=PERIOD, k_days_avg=K_DAYS_AVG)

    # Outputs
    out_png = Path("seasonal_naive_forecast.png")
    plot_last_14_days_with_forecast(s30, fc, out_png)

    out_csv = Path("forecast_output.csv")
    pd.DataFrame({"history": s30, "forecast": fc}).to_csv(out_csv, index_label="timestamp")
    print(f"[csv] saved → {out_csv.resolve()}")

if __name__ == "__main__":
    main()
