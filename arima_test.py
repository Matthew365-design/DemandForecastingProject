# pip install pandas numpy matplotlib openpyxl requests scikit-learn statsmodels pmdarima

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# ARIMA specific imports
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from pmdarima import auto_arima
    print("[arima] Successfully imported ARIMA libraries")
except ImportError as e:
    print(f"[error] Failed to import ARIMA libraries: {e}")
    print("Install with: pip install statsmodels pmdarima")
    exit(1)

# ---------------- CONFIG ----------------
EXCEL_PATH = Path("/mnt/c/Users/user/OneDrive/Documents/!DemandUROP/Data/demand_data.xlsx")
HISTORIC_SHEET = "HISTORIC"
SSEE_SHEET = "SSEE"
TIME_COL = "DATE/HOUR"

# time/grid settings
FREQ = "30min"
PERIOD = 48  # 48 half-hours per day
FORECAST_DAYS = 2
K_DAYS_AVG = 7  # Use 7-day average for baseline

# Weather variables to fetch
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "surface_pressure",
    "wind_speed_10m"
]

# ARIMA parameters
ARIMA_MAX_P = 3
ARIMA_MAX_D = 2
ARIMA_MAX_Q = 3
ARIMA_SEASONAL = True
ARIMA_M = PERIOD  # Seasonal period
ARIMA_MAX_P_SEASONAL = 2
ARIMA_MAX_D_SEASONAL = 1
ARIMA_MAX_Q_SEASONAL = 2
ARIMA_STEPWISE = True
ARIMA_SUPPRESS_WARNINGS = True
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
    df = df.drop('time', axis=1)
    return df.set_index("Datetime")

def load_substation_coords(excel_path: Path) -> pd.DataFrame:
    """Load substation coordinates from SSEE sheet"""
    coords_df = pd.read_excel(excel_path, sheet_name=SSEE_SHEET, engine="openpyxl")
    print(f"[coords] Available columns: {coords_df.columns.tolist()}")
    print(f"[coords] Loaded {len(coords_df)} substations")
    return coords_df

def get_substation_coords(coords_df: pd.DataFrame, substation_name: str) -> tuple:
    """Get lat/lon for a specific substation"""
    match = coords_df[coords_df['SUBSTATION'] == substation_name]
    if len(match) == 0:
        raise ValueError(f"Substation '{substation_name}' not found in SSEE sheet")
    lat = match.iloc[0]['X_EQ']  # X_EQ is latitude
    lon = match.iloc[0]['Y_EQ']  # Y_EQ is longitude
    return float(lat), float(lon)

def load_and_regularise(path: Path, substation_col: str) -> tuple:
    """Load and regularise demand data for specific substation with is_weekend column"""
    df = pd.read_excel(
        path,
        sheet_name=HISTORIC_SHEET,
        engine="openpyxl",
        usecols=[TIME_COL, substation_col, "is_weekend"],
        decimal=",",
        dtype={substation_col: "float64", "is_weekend": "bool"},
    )

    print(f"[debug] Loading substation: {substation_col}")
    print(f"[debug] columns: {list(df.columns)}")
    print(f"[debug] head:\n{df.head(3)}")

    # Parse timestamp and values
    t = pd.to_datetime(df[TIME_COL], errors="coerce", dayfirst=True)
    v = pd.to_numeric(df[substation_col], errors="coerce")
    w = df["is_weekend"].astype(bool)

    # Create series
    demand_series = pd.Series(v.values, index=t, name=substation_col).dropna().sort_index()
    weekend_series = pd.Series(w.values, index=t, name="is_weekend").sort_index()

    # Handle duplicates
    if not demand_series.index.is_unique:
        demand_series = demand_series.groupby(level=0).mean()
    if not weekend_series.index.is_unique:
        weekend_series = weekend_series.groupby(level=0).first()

    if len(demand_series) == 0:
        print(f"[warn] After parsing, series for {substation_col} is empty.")
        return demand_series, weekend_series

    # Regularise to 30-min grid
    demand_30min = demand_series.resample(FREQ).mean()
    weekend_30min = weekend_series.resample(FREQ).first()
    
    n_missing_before = int(demand_30min.isna().sum())
    demand_30min = demand_30min.interpolate(method="time").ffill().bfill().astype("float64")
    weekend_30min = weekend_30min.ffill().bfill()

    print(f"[regularise] points={len(demand_30min)}, filled_NaNs={n_missing_before}")
    return demand_30min, weekend_30min

def fetch_weather_for_timeframe(lat: float, lon: float, demand_series: pd.Series) -> pd.DataFrame:
    """Fetch weather data for the same timeframe as demand data"""
    start_date = demand_series.index.min().date()
    end_date = demand_series.index.max().date()
    
    print(f"[weather] Fetching for lat={lat}, lon={lon}")
    print(f"[weather] Date range: {start_date} to {end_date}")
    
    try:
        weather_df = get_weather(
            lat=lat, lon=lon, 
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            vars=WEATHER_VARS
        )
        weather_30min = weather_df.resample(FREQ).interpolate()
        print(f"[weather] Fetched {len(weather_30min)} weather points")
        return weather_30min
    except Exception as e:
        print(f"[error] Failed to fetch weather data: {e}")
        return pd.DataFrame()

def seasonal_naive_forecast(series: pd.Series, steps: int, period: int, k_days_avg: int = 0) -> pd.Series:
    """Seasonal naive forecast with k-day averaging"""
    last_stamp = series.index[-1]
    future_index = pd.date_range(start=last_stamp + pd.Timedelta(FREQ), periods=steps, freq=FREQ)

    if len(series) < period:
        print(f"[warn] Only {len(series)} points available (<{period}). Using last-value persistence.")
        return pd.Series(np.full(steps, series.iloc[-1], dtype=float), index=future_index, name="seasonal_naive")

    if k_days_avg <= 0:
        template = series[-period:].to_numpy()
    else:
        tail = series[-period * k_days_avg:]
        slots = np.arange(len(tail)) % period
        means = np.zeros(period, dtype=float)
        for slot in range(period):
            slot_values = tail[slots == slot]
            if len(slot_values) > 0:
                means[slot] = slot_values.mean()
            else:
                means[slot] = series.iloc[-1]
        template = means

    reps = int(np.ceil(steps / period))
    tiled = np.tile(template, reps)[:steps]
    return pd.Series(tiled, index=future_index, name="seasonal_naive")

def check_stationarity(series: pd.Series, name: str = "series") -> bool:
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    print(f"[stationarity] Checking stationarity for {name}...")
    
    # Perform ADF test
    result = adfuller(series.dropna())
    adf_statistic = result[0]
    p_value = result[1]
    critical_values = result[4]
    
    print(f"[stationarity] ADF Statistic: {adf_statistic:.6f}")
    print(f"[stationarity] p-value: {p_value:.6f}")
    print(f"[stationarity] Critical Values:")
    for key, value in critical_values.items():
        print(f"[stationarity]   {key}: {value:.3f}")
    
    # Determine if stationary
    is_stationary = p_value <= 0.05
    if is_stationary:
        print(f"[stationarity] {name} is STATIONARY (p-value <= 0.05)")
    else:
        print(f"[stationarity] {name} is NON-STATIONARY (p-value > 0.05)")
    
    return is_stationary

def prepare_arima_data(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for ARIMA modeling with exogenous variables"""
    print(f"[ARIMA] Preparing data with exogenous variables...")
    
    # Start with demand as the main series
    arima_data = pd.DataFrame({'demand': demand_series})
    
    # Add weekend feature
    weekend_aligned = weekend_series.reindex(demand_series.index).fillna(False).astype(float)
    arima_data['is_weekend'] = weekend_aligned
    
    # Add temperature if available
    if not weather_df.empty and 'temperature_2m' in weather_df.columns:
        temp_aligned = weather_df['temperature_2m'].reindex(demand_series.index)
        temp_aligned = temp_aligned.interpolate().fillna(method='ffill').fillna(method='bfill')
        arima_data['temperature_2m'] = temp_aligned
        print("[ARIMA] Added temperature_2m as exogenous variable")
    else:
        print("[ARIMA] No temperature data available")
    
    # Remove any rows with NaN values
    arima_data = arima_data.dropna()
    
    print(f"[ARIMA] Prepared dataset shape: {arima_data.shape}")
    print(f"[ARIMA] Columns: {list(arima_data.columns)}")
    print(f"[ARIMA] Date range: {arima_data.index.min()} to {arima_data.index.max()}")
    
    return arima_data

def auto_arima_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """Auto ARIMA forecast with exogenous variables"""
    print("[AUTO-ARIMA] Starting Auto ARIMA forecasting...")
    
    # Minimum data requirement
    min_data_points = max(10 * PERIOD, 500)  # At least 10 days or 500 points
    if len(demand_series) < min_data_points:
        print(f"[AUTO-ARIMA] Insufficient data ({len(demand_series)} points). Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        # Prepare data
        arima_data = prepare_arima_data(demand_series, weekend_series, weather_df)
        
        if len(arima_data) < min_data_points:
            print(f"[AUTO-ARIMA] Insufficient clean data ({len(arima_data)} points). Using seasonal naive fallback.")
            return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
        
        # Extract target and exogenous variables
        y = arima_data['demand']
        exog = arima_data.drop('demand', axis=1) if arima_data.shape[1] > 1 else None
        
        # Check stationarity
        is_stationary = check_stationarity(y, "demand")
        
        # Use recent data for faster training (last 30 days)
        recent_days = 30
        recent_points = recent_days * PERIOD
        if len(y) > recent_points:
            print(f"[AUTO-ARIMA] Using recent {recent_days} days ({recent_points} points) for faster training")
            y_train = y.tail(recent_points)
            exog_train = exog.tail(recent_points) if exog is not None else None
        else:
            y_train = y
            exog_train = exog
        
        print(f"[AUTO-ARIMA] Training on {len(y_train)} points")
        print(f"[AUTO-ARIMA] Exogenous variables: {list(exog_train.columns) if exog_train is not None else 'None'}")
        
        # Configure Auto ARIMA
        print("[AUTO-ARIMA] Running automatic model selection...")
        start_time = datetime.now()
        
        auto_model = auto_arima(
            y_train,
            exogenous=exog_train,
            start_p=0, start_q=0,
            max_p=ARIMA_MAX_P, max_q=ARIMA_MAX_Q, max_d=ARIMA_MAX_D,
            seasonal=ARIMA_SEASONAL,
            m=ARIMA_M,
            max_P=ARIMA_MAX_P_SEASONAL, max_Q=ARIMA_MAX_Q_SEASONAL, max_D=ARIMA_MAX_D_SEASONAL,
            stepwise=ARIMA_STEPWISE,
            suppress_warnings=ARIMA_SUPPRESS_WARNINGS,
            error_action='ignore',
            trace=True,
            n_jobs=-1  # Use all available cores
        )
        
        training_time = datetime.now() - start_time
        print(f"[AUTO-ARIMA] Model selection completed in {training_time.total_seconds():.2f} seconds")
        print(f"[AUTO-ARIMA] Best model: {auto_model.order} x {auto_model.seasonal_order}")
        print(f"[AUTO-ARIMA] AIC: {auto_model.aic():.2f}")
        
        # Prepare exogenous variables for forecasting
        if exog is not None:
            # Create future exogenous variables
            future_index = pd.date_range(
                start=y.index[-1] + pd.Timedelta(FREQ),
                periods=steps,
                freq=FREQ
            )
            
            # For weekend: use cyclical pattern
            future_weekend = []
            for i in range(steps):
                # Determine day of week for future timestamp
                future_date = future_index[i]
                is_weekend = future_date.weekday() >= 5  # Saturday=5, Sunday=6
                future_weekend.append(float(is_weekend))
            
            # For temperature: use seasonal pattern from last year if available
            future_temp = []
            if 'temperature_2m' in exog.columns:
                for i in range(steps):
                    # Use same time from previous year or week as fallback
                    future_date = future_index[i]
                    
                    # Try to find same day/hour from previous year
                    year_ago = future_date - pd.Timedelta(days=365)
                    week_ago = future_date - pd.Timedelta(days=7)
                    
                    temp_val = None
                    for reference_date in [year_ago, week_ago]:
                        closest_temp = exog['temperature_2m'].asof(reference_date)
                        if pd.notna(closest_temp):
                            temp_val = closest_temp
                            break
                    
                    if temp_val is None:
                        temp_val = exog['temperature_2m'].mean()  # Fallback to mean
                    
                    future_temp.append(temp_val)
            else:
                future_temp = [0.0] * steps  # Dummy values
            
            # Create future exogenous DataFrame
            future_exog_data = {
                'is_weekend': future_weekend,
                'temperature_2m': future_temp
            }
            future_exog = pd.DataFrame(future_exog_data, index=future_index)
            
            # Ensure column order matches training data
            future_exog = future_exog[exog.columns]
            
            print(f"[AUTO-ARIMA] Created future exogenous variables: {list(future_exog.columns)}")
        else:
            future_exog = None
        
        # Generate forecast
        print("[AUTO-ARIMA] Generating forecast...")
        forecast_result = auto_model.predict(n_periods=steps, exogenous=future_exog, return_conf_int=True)
        forecast_values = forecast_result[0]
        conf_int = forecast_result[1]
        
        # Create forecast series
        forecast_index = pd.date_range(
            start=y.index[-1] + pd.Timedelta(FREQ),
            periods=steps,
            freq=FREQ
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_index, name="auto_arima")
        
        # Print confidence intervals
        print(f"[AUTO-ARIMA] Forecast range: {forecast_values.min():.2f} to {forecast_values.max():.2f}")
        print(f"[AUTO-ARIMA] 95% Confidence Interval: [{conf_int[:, 0].mean():.2f}, {conf_int[:, 1].mean():.2f}]")
        
        print(f"[AUTO-ARIMA] Successfully generated {len(forecast_series)} forecasts")
        return forecast_series
        
    except Exception as e:
        print(f"[AUTO-ARIMA] Error in Auto ARIMA forecast: {e}. Using seasonal naive fallback.")
        import traceback
        traceback.print_exc()
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)

def simple_arima_forecast(demand_series: pd.Series, steps: int) -> pd.Series:
    """Simple ARIMA forecast without exogenous variables as fallback"""
    print("[SIMPLE-ARIMA] Running simple ARIMA forecast...")
    
    try:
        # Use recent data only for speed
        recent_points = min(30 * PERIOD, len(demand_series))
        y_train = demand_series.tail(recent_points)
        
        print(f"[SIMPLE-ARIMA] Training on {len(y_train)} recent points")
        
        # Simple automatic model selection
        model = auto_arima(
            y_train,
            start_p=0, start_q=0,
            max_p=2, max_q=2, max_d=2,
            seasonal=True, m=PERIOD,
            max_P=1, max_Q=1, max_D=1,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        print(f"[SIMPLE-ARIMA] Model: {model.order} x {model.seasonal_order}")
        
        # Generate forecast
        forecast_values = model.predict(n_periods=steps)
        
        # Create forecast series
        forecast_index = pd.date_range(
            start=demand_series.index[-1] + pd.Timedelta(FREQ),
            periods=steps,
            freq=FREQ
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_index, name="simple_arima")
        
        print(f"[SIMPLE-ARIMA] Successfully generated {len(forecast_series)} forecasts")
        return forecast_series
        
    except Exception as e:
        print(f"[SIMPLE-ARIMA] Error: {e}. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)

def calculate_metrics(actual: pd.Series, predicted: pd.Series):
    """Calculate MAPE, MSE, and wMAPE"""
    common_idx = actual.index.intersection(predicted.index)
    if len(common_idx) == 0:
        return {"MAPE": np.nan, "MSE": np.nan, "wMAPE": np.nan}
    
    y_true = actual.loc[common_idx].values
    y_pred = predicted.loc[common_idx].values
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) == 0:
        return {"MAPE": np.nan, "MSE": np.nan, "wMAPE": np.nan}
    
    mse = mean_squared_error(y_true, y_pred)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        mape = np.nan if np.isinf(mape) else mape
    
    wmape = np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    return {"MAPE": mape, "MSE": mse, "wMAPE": wmape}

def plot_forecasts_comparison(demand_series: pd.Series, seasonal_forecast: pd.Series, 
                            arima_forecast: pd.Series, simple_arima_forecast: pd.Series,
                            weather_df: pd.DataFrame, substation: str, out_path: Path):
    """Plot all forecasts with weather context"""
    lookback = 14 * PERIOD
    tail = demand_series.iloc[-lookback:] if len(demand_series) > lookback else demand_series
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Demand and forecasts
    tail.plot(ax=axes[0], label="Historical Demand", color="blue", linewidth=1.5)
    seasonal_forecast.plot(ax=axes[0], label=f"Seasonal Naive ({len(seasonal_forecast)} steps)", 
                          color="red", linewidth=2, linestyle='--')
    arima_forecast.plot(ax=axes[0], label=f"Auto ARIMA ({len(arima_forecast)} steps)", 
                       color="green", linewidth=2)
    simple_arima_forecast.plot(ax=axes[0], label=f"Simple ARIMA ({len(simple_arima_forecast)} steps)", 
                              color="orange", linewidth=2, linestyle=':')
    axes[0].set_title(f"{substation}: ARIMA Forecast Comparison")
    axes[0].set_ylabel("Demand")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Temperature context
    if not weather_df.empty and "temperature_2m" in weather_df.columns:
        weather_tail = weather_df.loc[tail.index[0]:arima_forecast.index[-1], "temperature_2m"]
        weather_tail.plot(ax=axes[1], label="Temperature", color="purple", linewidth=1)
        axes[1].set_title("Temperature Context")
        axes[1].set_ylabel("Temperature (°C)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No weather data available", ha="center", va="center", 
                    transform=axes[1].transAxes, fontsize=12)
        axes[1].set_title("Weather Data")
    
    # Plot 3: Forecast comparison (zoomed)
    forecast_start = seasonal_forecast.index[0]
    recent_history = demand_series.loc[forecast_start - pd.Timedelta(days=3):].tail(144)
    
    if len(recent_history) > 0:
        recent_history.plot(ax=axes[2], label="Recent History", color="blue", linewidth=1.5)
    seasonal_forecast.plot(ax=axes[2], label="Seasonal Naive", color="red", linewidth=2, linestyle='--')
    arima_forecast.plot(ax=axes[2], label="Auto ARIMA", color="green", linewidth=2)
    simple_arima_forecast.plot(ax=axes[2], label="Simple ARIMA", color="orange", linewidth=2, linestyle=':')
    axes[2].set_title("Forecast Detail View - ARIMA Models")
    axes[2].set_ylabel("Demand")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved → {out_path.resolve()}")

def create_metrics_table(seasonal_metrics: dict, arima_metrics: dict, simple_arima_metrics: dict, substation: str) -> str:
    """Create a formatted metrics comparison table"""
    table = f"""
{'='*90}
ARIMA FORECAST PERFORMANCE COMPARISON - {substation}
{'='*90}

{'Metric':<12} {'Seasonal Naive':<15} {'Auto ARIMA':<15} {'Simple ARIMA':<15} {'Best Model':<12}
{'-'*90}
"""
    
    for metric in ['MAPE', 'MSE', 'wMAPE']:
        seasonal_val = seasonal_metrics.get(metric, np.nan)
        arima_val = arima_metrics.get(metric, np.nan)
        simple_arima_val = simple_arima_metrics.get(metric, np.nan)
        
        values = {'Seasonal': seasonal_val, 'Auto ARIMA': arima_val, 'Simple ARIMA': simple_arima_val}
        valid_values = {k: v for k, v in values.items() if not np.isnan(v)}
        
        if valid_values:
            best_model = min(valid_values, key=valid_values.get)
        else:
            best_model = "N/A"
        
        seasonal_str = f"{seasonal_val:.3f}" if not np.isnan(seasonal_val) else "N/A"
        arima_str = f"{arima_val:.3f}" if not np.isnan(arima_val) else "N/A"
        simple_arima_str = f"{simple_arima_val:.3f}" if not np.isnan(simple_arima_val) else "N/A"
        
        table += f"{metric:<12} {seasonal_str:<15} {arima_str:<15} {simple_arima_str:<15} {best_model:<12}\n"
    
    table += f"\n{'='*90}\n"
    table += "Notes:\n"
    table += "- MAPE: Mean Absolute Percentage Error (lower is better)\n"
    table += "- MSE: Mean Squared Error (lower is better)\n"
    table += "- wMAPE: Weighted Mean Absolute Percentage Error (lower is better)\n"
    table += "- Auto ARIMA: Automatic model selection with exogenous variables\n"
    table += "- Simple ARIMA: Basic ARIMA without exogenous variables\n"
    
    return table

def main():
    print(f"[load] {EXCEL_PATH.resolve()}")
    
    # Load substation coordinates
    coords_df = load_substation_coords(EXCEL_PATH)
    available_substations = coords_df['SUBSTATION'].tolist()
    print(f"[info] Available substations: {available_substations[:10]}...")
    
    # Get user input
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
    
    # Load demand data and weekend data
    demand_series, weekend_series = load_and_regularise(EXCEL_PATH, substation_name)
    if demand_series.empty:
        print("[error] No demand data loaded")
        return
    
    print(f"[data] Loaded {len(demand_series):,} demand points and {len(weekend_series):,} weekend points")
    
    # Fetch weather data
    weather_df = fetch_weather_for_timeframe(lat, lon, demand_series)
    
    # Generate forecasts
    steps = FORECAST_DAYS * PERIOD
    print(f"\n[forecast] Generating {steps} step forecasts ({FORECAST_DAYS} days)...")
    
    # Baseline: Seasonal naive
    print("[forecast] Running seasonal naive baseline...")
    seasonal_forecast = seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    # Prepare validation data for metrics
    print("[forecast] Preparing validation data for metrics...")
    validation_start = demand_series.index[-7*PERIOD]
    validation_data = demand_series.loc[validation_start:]
    validation_weekend = weekend_series.loc[validation_start:]
    
    # Generate forecasts for validation period
    train_data = demand_series.loc[:validation_start - pd.Timedelta(FREQ)]
    train_weekend = weekend_series.loc[:validation_start - pd.Timedelta(FREQ)]
    
    print("[forecast] Running models for validation metrics...")
    val_seasonal = seasonal_naive_forecast(train_data, len(validation_data), PERIOD, K_DAYS_AVG)
    val_arima = auto_arima_forecast(train_data, train_weekend, weather_df.loc[:validation_start], len(validation_data))
    val_simple_arima = simple_arima_forecast(train_data, len(validation_data))
    
    # Calculate metrics
    seasonal_metrics = calculate_metrics(validation_data, val_seasonal)
    arima_metrics = calculate_metrics(validation_data, val_arima)
    simple_arima_metrics = calculate_metrics(validation_data, val_simple_arima)
    
    # Main forecasts
    print("[forecast] Running Auto ARIMA for main forecast...")
    arima_forecast_main = auto_arima_forecast(demand_series, weekend_series, weather_df, steps)
    
    print("[forecast] Running Simple ARIMA for main forecast...")
    simple_arima_forecast_main = simple_arima_forecast(demand_series, steps)
    
    # Create outputs
    safe_name = substation_name.replace("/", "_").replace("\\", "_")
    out_png = Path(f"forecast_comparison_arima_{safe_name}.png")
    plot_forecasts_comparison(demand_series, seasonal_forecast, arima_forecast_main, 
                            simple_arima_forecast_main, weather_df, substation_name, out_png)
    
    # Create and display metrics table
    metrics_table = create_metrics_table(seasonal_metrics, arima_metrics, simple_arima_metrics, substation_name)
    print(metrics_table)
    
    # Save results
    out_csv = Path(f"forecast_results_arima_{safe_name}.csv")
    output_data = {
        "demand_history": demand_series, 
        "seasonal_naive_forecast": seasonal_forecast,
        "auto_arima_forecast": arima_forecast_main,
        "simple_arima_forecast": simple_arima_forecast_main
    }
    
    if not weather_df.empty:
        for col in weather_df.columns:
            output_data[f"weather_{col}"] = weather_df[col]
    
    pd.DataFrame(output_data).to_csv(out_csv, index_label="timestamp")
    print(f"[csv] Results saved → {out_csv.resolve()}")
    
    # Save metrics to file
    metrics_file = Path(f"metrics_arima_{safe_name}.txt")
    with open(metrics_file, 'w') as f:
        f.write(metrics_table)
    print(f"[metrics] Metrics saved → {metrics_file.resolve()}")

if __name__ == "__main__":
    main()