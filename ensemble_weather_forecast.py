# pip install pandas numpy matplotlib openpyxl requests

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta


EXCEL_PATH = Path(r"C:\Users\user\OneDrive\Documents\!DemandUROP\Data\demand_data.xlsx")
HISTORIC_SHEET = "HISTORIC"
SSEE_SHEET = "SSEE"
TIME_COL = "DATE/HOUR"

# time/grid settings
FREQ = "30min"
PERIOD = 48  # 48 half-hours per day
FORECAST_DAYS = 2
K_DAYS_AVG = 0

# Weather variables to fetch
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "surface_pressure",
    "wind_speed_10m"
]
# ----------------------------------------

def get_weather(lat, lon, start, end, vars):
    """Fetch weather data from Open-Meteo API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(vars),
        "timezone": "auto"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data["hourly"])
    df["Datetime"] = pd.to_datetime(df["time"])
    return df.set_index("Datetime")

def load_substation_coords(excel_path: Path) -> pd.DataFrame:
    """Load substation coordinates from SSEE sheet"""
    coords_df = pd.read_excel(
        excel_path,
        sheet_name=SSEE_SHEET,
        engine="openpyxl"
    )
    print(f"[coords] Available columns: {coords_df.columns.tolist()}")
    print(f"[coords] Loaded {len(coords_df)} substations")
    return coords_df

def get_substation_coords(coords_df: pd.DataFrame, substation_name: str) -> tuple:
    """Get lat/lon for a specific substation"""
    # Updated column names: 'SUBSTATION', 'X_EQ', 'Y_EQ'
    match = coords_df[coords_df['SUBSTATION'] == substation_name]
    if len(match) == 0:
        raise ValueError(f"Substation '{substation_name}' not found in SSEE sheet")
    
    lat = match.iloc[0]['Y_EQ']  # Y_EQ is latitude
    lon = match.iloc[0]['X_EQ']  # X_EQ is longitude
    return float(lat), float(lon)

def load_and_regularise(path: Path, substation_col: str) -> pd.Series:
    """Load and regularise demand data for specific substation"""
    df = pd.read_excel(
        path,
        sheet_name=HISTORIC_SHEET,
        engine="openpyxl",
        usecols=[TIME_COL, substation_col],
        decimal=",",
        dtype={substation_col: "float64"},
    )

    print(f"[debug] Loading substation: {substation_col}")
    print(f"[debug] columns: {list(df.columns)}")
    print(f"[debug] head:\n{df.head(3)}")

    # Parse timestamp and values
    t = pd.to_datetime(df[TIME_COL], errors="coerce", dayfirst=True)
    v = pd.to_numeric(df[substation_col], errors="coerce")

    s = pd.Series(v.values, index=t, name=substation_col).dropna().sort_index()

    # Handle duplicates
    if not s.index.is_unique:
        s = s.groupby(level=0).mean()

    if len(s) == 0:
        print(f"[warn] After parsing, series for {substation_col} is empty.")
        return s

    # Regularise to 30-min grid
    s30 = s.resample(FREQ).mean()
    n_missing_before = int(s30.isna().sum())
    s30 = s30.interpolate(method="time").ffill().bfill().astype("float64")

    print(f"[regularise] points={len(s30)}, filled_NaNs={n_missing_before}")
    return s30

def fetch_weather_for_timeframe(lat: float, lon: float, demand_series: pd.Series) -> pd.DataFrame:
    """Fetch weather data for the same timeframe as demand data"""
    start_date = demand_series.index.min().date()
    end_date = demand_series.index.max().date()
    
    print(f"[weather] Fetching for lat={lat}, lon={lon}")
    print(f"[weather] Date range: {start_date} to {end_date}")
    
    try:
        weather_df = get_weather(
            lat=lat,
            lon=lon, 
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            vars=WEATHER_VARS
        )
        
        # Resample to 30-min to match demand data
        weather_30min = weather_df.resample(FREQ).interpolate()
        print(f"[weather] Fetched {len(weather_30min)} weather points")
        return weather_30min
        
    except Exception as e:
        print(f"[error] Failed to fetch weather data: {e}")
        return pd.DataFrame()

def seasonal_naive_forecast(series: pd.Series, steps: int, period: int, k_days_avg: int = 0) -> pd.Series:
    """Same seasonal naive forecast as original"""
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

def simple_ensemble_forecast(demand_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """
    Simple ensemble combining seasonal naive with weather-influenced adjustments
    This is a placeholder - more sophisticated ensemble methods can be added later
    """
    # Base seasonal naive forecast
    base_forecast = seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    if weather_df.empty:
        print("[ensemble] No weather data available, using pure seasonal naive")
        return base_forecast
    
    # For now, just return the base forecast
    # TODO: Add weather-based adjustments here
    print("[ensemble] Using seasonal naive base (weather integration pending)")
    return base_forecast

def plot_forecast_with_weather(demand_series: pd.Series, forecast: pd.Series, weather_df: pd.DataFrame, substation: str, out_path: Path):
    """Plot demand forecast with weather context"""
    lookback = 14 * PERIOD
    tail = demand_series.iloc[-lookback:] if len(demand_series) > lookback else demand_series
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot demand and forecast
    tail.plot(ax=ax1, label="demand history", color="blue")
    forecast.plot(ax=ax1, label=f"forecast (+{len(forecast)} steps)", color="red")
    ax1.set_title(f"{substation}: Demand Forecast")
    ax1.set_ylabel("Demand")
    ax1.legend()
    
    # Plot temperature if available
    if not weather_df.empty and "temperature_2m" in weather_df.columns:
        weather_tail = weather_df.loc[tail.index[0]:forecast.index[-1], "temperature_2m"]
        weather_tail.plot(ax=ax2, label="temperature", color="green")
        ax2.set_title("Temperature Context")
        ax2.set_ylabel("Temperature (°C)")
        ax2.legend()
    else:
        ax2.text(0.5, 0.5, "No weather data available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Weather Data")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[plot] saved → {out_path.resolve()}")

def main():
    # Get user input for substation
    print(f"[load] {EXCEL_PATH.resolve()}")
    
    # Load substation coordinates
    coords_df = load_substation_coords(EXCEL_PATH)
    
    # Show available substations
    available_substations = coords_df['SUBSTATION'].tolist()  # Updated column name
    print(f"[info] Available substations: {available_substations[:10]}...")  # Show first 10
    
    # Get user input (you can modify this to accept command line args)
    substation_name = input("Enter substation name: ").strip()
    
    if substation_name not in available_substations:
        print(f"[error] Substation '{substation_name}' not found")
        return
    
    # Get coordinates
    try:
        lat, lon = get_substation_coords(coords_df, substation_name)
        print(f"[coords] {substation_name}: lat={lat}, lon={lon}")
    except Exception as e:
        print(f"[error] {e}")
        return
    
    # Load demand data
    demand_series = load_and_regularise(EXCEL_PATH, substation_name)
    if demand_series.empty:
        print("[error] No demand data loaded")
        return
    
    # Fetch weather data
    weather_df = fetch_weather_for_timeframe(lat, lon, demand_series)
    
    # Generate ensemble forecast
    steps = FORECAST_DAYS * PERIOD
    forecast = simple_ensemble_forecast(demand_series, weather_df, steps)
    
    # Create outputs
    safe_name = substation_name.replace("/", "_").replace("\\", "_")
    out_png = Path(f"ensemble_forecast_{safe_name}.png")
    plot_forecast_with_weather(demand_series, forecast, weather_df, substation_name, out_png)
    
    # Save data
    out_csv = Path(f"ensemble_output_{safe_name}.csv")
    output_data = {"demand_history": demand_series, "forecast": forecast}
    
    # Add weather data if available
    if not weather_df.empty:
        for col in weather_df.columns:
            output_data[f"weather_{col}"] = weather_df[col]
    
    pd.DataFrame(output_data).to_csv(out_csv, index_label="timestamp")
    print(f"[csv] saved → {out_csv.resolve()}")

if __name__ == "__main__":
    main()