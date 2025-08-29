# pip install pandas numpy matplotlib openpyxl requests tensorflow scikit-learn darts[torch]

# GPU Memory Optimization and Setup
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"     # quiet logs
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    try: 
        tf.config.experimental.set_memory_growth(g, True)
        # Force higher GPU utilization
        tf.config.experimental.set_virtual_device_configuration(
            g, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]  # Use 6GB
        )
    except: pass

from keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# Darts imports for N-BEATS
try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
    from darts.utils.utils import ModelMode
    import torch
    print(f"[darts] Successfully imported Darts with PyTorch backend")
    print(f"[torch] PyTorch version: {torch.__version__}")
    print(f"[torch] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[torch] CUDA device: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"[error] Failed to import Darts: {e}")
    print("Install with: pip install darts[torch]")
    exit(1)

# TensorFlow/Keras imports for LSTM
try:
    import keras
    from keras.models import Sequential
    from keras.layers import LSTM, Dense, Dropout
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    print(f"[tensorflow] Using TensorFlow {tf.__version__} with Keras {keras.__version__}")
except ImportError:
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        print(f"[tensorflow] Using TensorFlow {tf.__version__} with built-in Keras")
    except ImportError as e:
        print(f"[error] Cannot import Keras: {e}")
        exit(1)

import warnings
warnings.filterwarnings('ignore')

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

# Model parameters - OPTIMIZED FOR SPEED
LSTM_LOOKBACK = 3 * PERIOD  # 144 points instead of 336
LSTM_EPOCHS = 20  # Reduced epochs
LSTM_BATCH_SIZE = 512  # Larger batches

# N-BEATS parameters for Darts
NBEATS_INPUT_CHUNK_LENGTH = 7 * PERIOD  # 7 days lookback
NBEATS_OUTPUT_CHUNK_LENGTH = FORECAST_DAYS * PERIOD  # 2 days forecast
NBEATS_EPOCHS = 100
NBEATS_BATCH_SIZE = 256
NBEATS_NUM_STACKS = 30
NBEATS_NUM_BLOCKS = 1
NBEATS_NUM_LAYERS = 4
NBEATS_LAYER_WIDTHS = 512
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
            lat=lat,
            lon=lon, 
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

def prepare_lstm_data(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, lookback: int):
    """Prepare data for LSTM training - OPTIMIZED FOR SPEED"""
    print(f"[LSTM] Using 3 features: demand, temperature_2m, is_weekend")
    
    # CRITICAL: Limit data to recent history for faster training
    cutoff_date = demand_series.index[-1] - pd.Timedelta(days=90)  # Only use last 90 days
    recent_demand = demand_series.loc[cutoff_date:]
    recent_weekend = weekend_series.loc[cutoff_date:]
    
    print(f"[LSTM] LIMITED to recent {len(recent_demand)} points (last 90 days) for faster training")
    
    # Start with demand data
    features_dict = {'demand': recent_demand}
    
    # Add weekend feature
    features_dict['is_weekend'] = recent_weekend.astype(float)
    
    # Add temperature if available - also limit to recent data
    if not weather_df.empty and 'temperature_2m' in weather_df.columns:
        recent_weather = weather_df.loc[cutoff_date:]
        common_index = recent_demand.index.intersection(recent_weather.index)
        if len(common_index) > 0:
            temp_aligned = recent_weather.loc[common_index, 'temperature_2m']
            features_dict['temperature_2m'] = temp_aligned
            print("[LSTM] Added temperature_2m feature")
        else:
            features_dict['temperature_2m'] = pd.Series(0.0, index=recent_demand.index)
    else:
        features_dict['temperature_2m'] = pd.Series(0.0, index=recent_demand.index)
    
    # Combine all features
    features = pd.DataFrame(features_dict)
    common_index = features.dropna().index
    features = features.loc[common_index]
    
    print(f"[LSTM] Feature matrix shape: {features.shape}")
    
    if len(features) < lookback + 50:
        raise ValueError(f"Insufficient data: {len(features)} points, need at least {lookback + 50}")
    
    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # CRITICAL: Create sequences - limit to avoid memory issues
    max_sequences = min(2000, len(scaled_features) - lookback)  # Cap at 2000 sequences
    X, y = [], []
    
    start_idx = max(lookback, len(scaled_features) - max_sequences)
    for i in range(start_idx, len(scaled_features)):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_features[i, 0])
    
    print(f"[LSTM] Created {len(X)} sequences (LIMITED for speed)")
    return np.array(X), np.array(y), scaler, features.columns

def build_lstm_model(input_shape):
    """Build optimized LSTM model for higher GPU usage"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape, dropout=0.2, recurrent_dropout=0.2),
            LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
            LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, dtype='float32')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse',
            metrics=['mae']
        )
        
        print(f"[LSTM] Model created with input shape: {input_shape}")
        print(f"[LSTM] Model parameters: {model.count_params():,}")
        return model

def prepare_darts_data(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame) -> tuple:
    """Prepare data for Darts N-BEATS model"""
    print(f"[NBEATS] Preparing Darts TimeSeries data...")
    
    # Create main demand TimeSeries
    demand_ts = TimeSeries.from_series(demand_series, freq=FREQ)
    print(f"[NBEATS] Created demand TimeSeries: {len(demand_ts)} points")
    
    # Prepare covariates (weekend and temperature)
    covariates_dict = {}
    
    # Add weekend covariate
    weekend_aligned = weekend_series.reindex(demand_series.index).fillna(False).astype(float)
    covariates_dict['is_weekend'] = weekend_aligned
    
    # Add temperature covariate if available
    if not weather_df.empty and 'temperature_2m' in weather_df.columns:
        temp_aligned = weather_df['temperature_2m'].reindex(demand_series.index)
        temp_aligned = temp_aligned.interpolate().fillna(method='ffill').fillna(method='bfill')
        covariates_dict['temperature_2m'] = temp_aligned
        print("[NBEATS] Added temperature_2m covariate")
    else:
        print("[NBEATS] No temperature data, using demand + weekend only")
        covariates_dict['temperature_2m'] = pd.Series(0.0, index=demand_series.index)
    
    # Create covariates DataFrame and TimeSeries
    if covariates_dict:
        covariates_df = pd.DataFrame(covariates_dict)
        covariates_ts = TimeSeries.from_dataframe(
            covariates_df.reset_index(), 
            time_col=covariates_df.index.name or 'time',
            freq=FREQ
        )
        print(f"[NBEATS] Created covariates TimeSeries: {len(covariates_ts)} points, {covariates_ts.n_components} components")
        return demand_ts, covariates_ts
    else:
        return demand_ts, None

def lstm_ensemble_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """LSTM-based ensemble forecast with optimized GPU usage"""
    print("[LSTM] Preparing data...")
    
    if len(demand_series) < LSTM_LOOKBACK + 100:
        print(f"[LSTM] Insufficient data for LSTM training. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        X, y, scaler, feature_cols = prepare_lstm_data(demand_series, weekend_series, weather_df, LSTM_LOOKBACK)
        
        if len(X) < 100:
            print(f"[LSTM] Too few training samples ({len(X)}). Using seasonal naive fallback.")
            return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
        
        # Split into train/test
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"[LSTM] Training on {len(X_train):,} samples, testing on {len(X_test):,} samples")
        print(f"[LSTM] Input shape: {X_train.shape}")
        
        # Force training on GPU
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            print("[LSTM] Building optimized model on GPU...")
            model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
            ]
            
            print("[LSTM] Starting optimized GPU training...")
            start_time = datetime.now()
            
            # Train - removed problematic multiprocessing args
            history = model.fit(X_train, y_train, 
                               epochs=LSTM_EPOCHS, 
                               batch_size=LSTM_BATCH_SIZE, 
                               validation_data=(X_test, y_test),
                               callbacks=callbacks,
                               verbose=1)
            
            training_time = datetime.now() - start_time
            print(f"[LSTM] Training completed in {training_time.total_seconds():.2f} seconds")
        
        # Generate forecasts on GPU
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            forecasts = []
            
            print("[LSTM] Generating forecasts on GPU...")
            for step in range(steps):
                pred = model.predict(last_sequence, verbose=0)[0, 0]
                forecasts.append(pred)
                
                new_row = last_sequence[0, -1].copy()
                new_row[0] = pred
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1] = new_row
        
        # Inverse transform forecasts
        dummy_features = np.zeros((len(forecasts), len(feature_cols)))
        dummy_features[:, 0] = forecasts
        forecasts_scaled = scaler.inverse_transform(dummy_features)[:, 0]
        
        # Create forecast series
        last_stamp = demand_series.index[-1]
        future_index = pd.date_range(start=last_stamp + pd.Timedelta(FREQ), periods=steps, freq=FREQ)
        
        print(f"[LSTM] Successfully generated {len(forecasts_scaled)} forecasts using optimized GPU")
        return pd.Series(forecasts_scaled, index=future_index, name="lstm_ensemble")
        
    except Exception as e:
        print(f"[LSTM] Error in LSTM forecast: {e}. Using seasonal naive fallback.")
        import traceback
        traceback.print_exc()
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)

def nbeats_darts_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """N-BEATS forecast using Darts library with GPU support"""
    print("[NBEATS] Preparing Darts N-BEATS forecast...")
    
    if len(demand_series) < NBEATS_INPUT_CHUNK_LENGTH + 100:
        print(f"[NBEATS] Insufficient data for N-BEATS training. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        # Prepare Darts TimeSeries data
        demand_ts, covariates_ts = prepare_darts_data(demand_series, weekend_series, weather_df)
        
        # Split into train/test for validation
        train_size = int(len(demand_ts) * 0.8)
        train_ts = demand_ts[:train_size]
        test_ts = demand_ts[train_size:]
        
        if covariates_ts is not None:
            train_cov = covariates_ts[:train_size]
            test_cov = covariates_ts[train_size:]
        else:
            train_cov = None
            test_cov = None
        
        print(f"[NBEATS] Training on {len(train_ts)} points, testing on {len(test_ts)} points")
        
        # Configure N-BEATS model with GPU support
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[NBEATS] Using device: {device}")
        
        model = NBEATSModel(
            input_chunk_length=NBEATS_INPUT_CHUNK_LENGTH,
            output_chunk_length=NBEATS_OUTPUT_CHUNK_LENGTH,
            num_stacks=NBEATS_NUM_STACKS,
            num_blocks=NBEATS_NUM_BLOCKS,
            num_layers=NBEATS_NUM_LAYERS,
            layer_widths=NBEATS_LAYER_WIDTHS,
            n_epochs=NBEATS_EPOCHS,
            batch_size=NBEATS_BATCH_SIZE,
            model_name="demand_nbeats",
            force_reset=True,
            save_checkpoints=False,
            generic_architecture=True,  # Use generic architecture for flexibility
            optimizer_kwargs={'lr': 1e-3},
            lr_scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            lr_scheduler_kwargs={'patience': 10, 'factor': 0.5},
            random_state=42,
            pl_trainer_kwargs={
                "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
                "devices": 1 if torch.cuda.is_available() else None,
                "precision": "16-mixed" if torch.cuda.is_available() else 32,
                "enable_progress_bar": True,
                "enable_model_summary": True,
            }
        )
        
        print("[NBEATS] Starting N-BEATS training on GPU...")
        start_time = datetime.now()
        
        # Train the model
        if covariates_ts is not None:
            model.fit(train_ts, past_covariates=train_cov, verbose=True)
        else:
            model.fit(train_ts, verbose=True)
        
        training_time = datetime.now() - start_time
        print(f"[NBEATS] Training completed in {training_time.total_seconds():.2f} seconds")
        
        # Generate forecast
        print("[NBEATS] Generating forecast...")
        if covariates_ts is not None:
            # Extend covariates for forecast period
            last_cov_values = covariates_ts.tail(NBEATS_INPUT_CHUNK_LENGTH)
            # For future covariates, we'll use the pattern from the last week
            future_cov_data = []
            for i in range(steps):
                # Use cyclical pattern for weekend (weekly seasonality)
                week_position = i % (7 * PERIOD)
                historical_position = len(covariates_ts) - (7 * PERIOD) + week_position
                if historical_position >= 0:
                    future_cov_data.append(covariates_ts.values()[historical_position])
                else:
                    future_cov_data.append(covariates_ts.values()[-1])
            
            # Create future covariates TimeSeries
            future_index = pd.date_range(
                start=demand_series.index[-1] + pd.Timedelta(FREQ),
                periods=steps,
                freq=FREQ
            )
            future_cov_df = pd.DataFrame(future_cov_data, 
                                       index=future_index,
                                       columns=['is_weekend', 'temperature_2m'])
            future_cov_ts = TimeSeries.from_dataframe(
                future_cov_df.reset_index(),
                time_col='index',
                freq=FREQ
            )
            
            full_cov = covariates_ts.append(future_cov_ts)
            forecast_ts = model.predict(n=steps, past_covariates=full_cov)
        else:
            forecast_ts = model.predict(n=steps)
        
        # Convert forecast back to pandas Series
        forecast_values = forecast_ts.values().flatten()
        forecast_index = pd.date_range(
            start=demand_series.index[-1] + pd.Timedelta(FREQ),
            periods=len(forecast_values),
            freq=FREQ
        )
        
        forecast_series = pd.Series(forecast_values, index=forecast_index, name="nbeats_darts")
        
        print(f"[NBEATS] Successfully generated {len(forecast_series)} forecasts using Darts N-BEATS")
        return forecast_series
        
    except Exception as e:
        print(f"[NBEATS] Error in Darts N-BEATS forecast: {e}. Using seasonal naive fallback.")
        import traceback
        traceback.print_exc()
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
                            lstm_forecast: pd.Series, nbeats_forecast: pd.Series, 
                            weather_df: pd.DataFrame, substation: str, out_path: Path):
    """Plot all three forecasts with weather context and metrics"""
    lookback = 14 * PERIOD
    tail = demand_series.iloc[-lookback:] if len(demand_series) > lookback else demand_series
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # Plot 1: Demand and forecasts
    tail.plot(ax=axes[0], label="Historical Demand", color="blue", linewidth=1.5)
    seasonal_forecast.plot(ax=axes[0], label=f"Seasonal Naive ({len(seasonal_forecast)} steps)", 
                          color="red", linewidth=2, linestyle='--')
    lstm_forecast.plot(ax=axes[0], label=f"LSTM Ensemble ({len(lstm_forecast)} steps)", 
                      color="purple", linewidth=2)
    nbeats_forecast.plot(ax=axes[0], label=f"N-BEATS Darts ({len(nbeats_forecast)} steps)", 
                        color="orange", linewidth=2)
    axes[0].set_title(f"{substation}: Demand Forecast Comparison (3 Models)")
    axes[0].set_ylabel("Demand")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Temperature context
    if not weather_df.empty and "temperature_2m" in weather_df.columns:
        weather_tail = weather_df.loc[tail.index[0]:nbeats_forecast.index[-1], "temperature_2m"]
        weather_tail.plot(ax=axes[1], label="Temperature", color="green", linewidth=1)
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
    lstm_forecast.plot(ax=axes[2], label="LSTM Ensemble", color="purple", linewidth=2)
    nbeats_forecast.plot(ax=axes[2], label="N-BEATS Darts", color="orange", linewidth=2)
    axes[2].set_title("Forecast Detail View - All Models")
    axes[2].set_ylabel("Demand")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved → {out_path.resolve()}")

def create_metrics_table(seasonal_metrics: dict, lstm_metrics: dict, nbeats_metrics: dict, substation: str) -> str:
    """Create a formatted metrics comparison table for all three models"""
    table = f"""
{'='*80}
FORECAST PERFORMANCE COMPARISON - {substation}
{'='*80}

{'Metric':<12} {'Seasonal Naive':<15} {'LSTM Ensemble':<15} {'N-BEATS Darts':<15} {'Best Model':<12}
{'-'*80}
"""
    
    for metric in ['MAPE', 'MSE', 'wMAPE']:
        seasonal_val = seasonal_metrics.get(metric, np.nan)
        lstm_val = lstm_metrics.get(metric, np.nan)
        nbeats_val = nbeats_metrics.get(metric, np.nan)
        
        values = {'Seasonal': seasonal_val, 'LSTM': lstm_val, 'N-BEATS': nbeats_val}
        valid_values = {k: v for k, v in values.items() if not np.isnan(v)}
        
        if valid_values:
            best_model = min(valid_values, key=valid_values.get)
        else:
            best_model = "N/A"
        
        seasonal_str = f"{seasonal_val:.3f}" if not np.isnan(seasonal_val) else "N/A"
        lstm_str = f"{lstm_val:.3f}" if not np.isnan(lstm_val) else "N/A"
        nbeats_str = f"{nbeats_val:.3f}" if not np.isnan(nbeats_val) else "N/A"
        
        table += f"{metric:<12} {seasonal_str:<15} {lstm_str:<15} {nbeats_str:<15} {best_model:<12}\n"
    
    table += f"\n{'='*80}\n"
    table += "Notes:\n"
    table += "- MAPE: Mean Absolute Percentage Error (lower is better)\n"
    table += "- MSE: Mean Squared Error (lower is better)\n"
    table += "- wMAPE: Weighted Mean Absolute Percentage Error (lower is better)\n"
    table += "- Best Model: Model with lowest error for each metric\n"
    table += "- N-BEATS: Using Darts library with PyTorch backend\n"
    
    return table

# Enhanced GPU Setup and Verification
print("\n[GPU] Checking GPU availability and optimizing...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[GPU] TensorFlow - Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
    print(f"[GPU] TensorFlow - CUDA built: {tf.test.is_built_with_cuda()}")
    
    try:
        tf.config.experimental.set_synchronous_execution(False)
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
        tf.config.optimizer.set_jit(True)
        tf.config.experimental.enable_tensor_float_32_execution(True)
        print("[GPU] TensorFlow - Applied enhanced performance optimizations")
    except Exception as e:
        print(f"[GPU] TensorFlow - Could not apply some optimizations: {e}")
else:
    print("[GPU] TensorFlow - No GPUs found - will use CPU")

# PyTorch GPU check
if torch.cuda.is_available():
    print(f"[GPU] PyTorch - CUDA available: {torch.cuda.get_device_name(0)}")
    print(f"[GPU] PyTorch - CUDA version: {torch.version.cuda}")
else:
    print("[GPU] PyTorch - No CUDA support, will use CPU")

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
    val_lstm = lstm_ensemble_forecast(train_data, train_weekend, weather_df.loc[:validation_start], len(validation_data))
    val_nbeats = nbeats_darts_forecast(train_data, train_weekend, weather_df.loc[:validation_start], len(validation_data))
    
    # Calculate metrics
    seasonal_metrics = calculate_metrics(validation_data, val_seasonal)
    lstm_metrics = calculate_metrics(validation_data, val_lstm)
    nbeats_metrics = calculate_metrics(validation_data, val_nbeats)
    
    # Main forecasts
    print("[forecast] Running LSTM ensemble for main forecast...")
    lstm_forecast = lstm_ensemble_forecast(demand_series, weekend_series, weather_df, steps)
    
    print("[forecast] Running Darts N-BEATS for main forecast...")
    nbeats_forecast_main = nbeats_darts_forecast(demand_series, weekend_series, weather_df, steps)
    
    # Create outputs
    safe_name = substation_name.replace("/", "_").replace("\\", "_")
    out_png = Path(f"forecast_comparison_darts_{safe_name}.png")
    plot_forecasts_comparison(demand_series, seasonal_forecast, lstm_forecast, 
                            nbeats_forecast_main, weather_df, substation_name, out_png)
    
    # Create and display metrics table
    metrics_table = create_metrics_table(seasonal_metrics, lstm_metrics, nbeats_metrics, substation_name)
    print(metrics_table)
    
    # Save results
    out_csv = Path(f"forecast_results_darts_{safe_name}.csv")
    output_data = {
        "demand_history": demand_series, 
        "seasonal_naive_forecast": seasonal_forecast,
        "lstm_ensemble_forecast": lstm_forecast,
        "nbeats_darts_forecast": nbeats_forecast_main
    }
    
    if not weather_df.empty:
        for col in weather_df.columns:
            output_data[f"weather_{col}"] = weather_df[col]
    
    pd.DataFrame(output_data).to_csv(out_csv, index_label="timestamp")
    print(f"[csv] Results saved → {out_csv.resolve()}")
    
    # Save metrics to file
    metrics_file = Path(f"metrics_darts_{safe_name}.txt")
    with open(metrics_file, 'w') as f:
        f.write(metrics_table)
    print(f"[metrics] Metrics saved → {metrics_file.resolve()}")

if __name__ == "__main__":
    main()