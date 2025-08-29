# pip install pandas numpy matplotlib openpyxl requests tensorflow scikit-learn

# GPU Memory Optimization and Setup
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    try: 
        tf.config.experimental.set_memory_growth(g, True)
        tf.config.experimental.set_virtual_device_configuration(
            g, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
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
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports for Transformer
try:
    import keras
    from keras.models import Model
    from keras.layers import (
        Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
        GlobalAveragePooling1D, Conv1D, Add, Embedding, Concatenate
    )
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print(f"[tensorflow] Using TensorFlow {tf.__version__} with Keras {keras.__version__}")
except ImportError:
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
            GlobalAveragePooling1D, Conv1D, Add, Embedding, Concatenate
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        print(f"[tensorflow] Using TensorFlow {tf.__version__} with built-in Keras")
    except ImportError as e:
        print(f"[error] Cannot import Keras: {e}")
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
K_DAYS_AVG = 7

# Weather variables to fetch
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "surface_pressure",
    "wind_speed_10m"
]

# Transformer parameters
TRANSFORMER_LOOKBACK = 7 * PERIOD  # 7 days of history
TRANSFORMER_EPOCHS = 100
TRANSFORMER_BATCH_SIZE = 64  # Smaller batch for transformer
TRANSFORMER_D_MODEL = 128  # Model dimension
TRANSFORMER_NUM_HEADS = 8  # Attention heads
TRANSFORMER_NUM_LAYERS = 4  # Transformer layers
TRANSFORMER_DFF = 512  # Feed-forward dimension
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LEARNING_RATE = 1e-4

# TCA (Temporal Convolutional Attention) parameters
TCA_FILTERS = [64, 128, 256]  # Convolutional filters
TCA_KERNEL_SIZE = 3
TCA_DILATION_RATES = [1, 2, 4, 8]  # Dilated convolutions for temporal patterns
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
    lat = match.iloc[0]['X_EQ']
    lon = match.iloc[0]['Y_EQ']
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

def prepare_transformer_data(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, lookback: int):
    """Prepare data for Transformer training with temporal features"""
    print(f"[TRANSFORMER] Using enhanced features: demand, temperature_2m, is_weekend + temporal features")
    
    # Limit to recent data for faster training
    cutoff_date = demand_series.index[-1] - pd.Timedelta(days=120)  # 4 months
    recent_demand = demand_series.loc[cutoff_date:]
    recent_weekend = weekend_series.loc[cutoff_date:]
    
    print(f"[TRANSFORMER] Limited to recent {len(recent_demand)} points (last 120 days)")
    
    # Create base features
    features_dict = {'demand': recent_demand}
    features_dict['is_weekend'] = recent_weekend.astype(float)
    
    # Add weather features
    if not weather_df.empty and 'temperature_2m' in weather_df.columns:
        recent_weather = weather_df.loc[cutoff_date:]
        common_index = recent_demand.index.intersection(recent_weather.index)
        if len(common_index) > 0:
            temp_aligned = recent_weather.loc[common_index, 'temperature_2m']
            features_dict['temperature_2m'] = temp_aligned
            
            # Add more weather features for transformer
            if 'relative_humidity_2m' in recent_weather.columns:
                humidity_aligned = recent_weather.loc[common_index, 'relative_humidity_2m']
                features_dict['humidity'] = humidity_aligned
            
        print("[TRANSFORMER] Added weather features")
    else:
        features_dict['temperature_2m'] = pd.Series(0.0, index=recent_demand.index)
        features_dict['humidity'] = pd.Series(0.0, index=recent_demand.index)
    
    # Add temporal features for transformer
    timestamps = recent_demand.index
    features_dict['hour_of_day'] = [t.hour + t.minute/60.0 for t in timestamps]
    features_dict['day_of_week'] = [float(t.weekday()) for t in timestamps]
    features_dict['day_of_year'] = [float(t.dayofyear) for t in timestamps]
    
    # Add cyclical features
    features_dict['hour_sin'] = np.sin(2 * np.pi * np.array(features_dict['hour_of_day']) / 24)
    features_dict['hour_cos'] = np.cos(2 * np.pi * np.array(features_dict['hour_of_day']) / 24)
    features_dict['day_sin'] = np.sin(2 * np.pi * np.array(features_dict['day_of_week']) / 7)
    features_dict['day_cos'] = np.cos(2 * np.pi * np.array(features_dict['day_of_week']) / 7)
    
    # Combine features
    features = pd.DataFrame(features_dict)
    features = features.dropna()
    
    print(f"[TRANSFORMER] Feature matrix shape: {features.shape}")
    print(f"[TRANSFORMER] Features: {list(features.columns)}")
    
    if len(features) < lookback + 100:
        raise ValueError(f"Insufficient data: {len(features)} points, need at least {lookback + 100}")
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences with limited data for speed
    max_sequences = min(3000, len(scaled_features) - lookback)
    X, y = [], []
    
    start_idx = max(lookback, len(scaled_features) - max_sequences)
    for i in range(start_idx, len(scaled_features)):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_features[i, 0])  # demand is first column
    
    print(f"[TRANSFORMER] Created {len(X)} sequences")
    return np.array(X), np.array(y), scaler, features.columns

def positional_encoding(length, d_model):
    """Create positional encoding for transformer"""
    angle_rads = np.arange(length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_tca_block(inputs, filters, kernel_size, dilation_rate, dropout_rate, block_id):
    """Create a Temporal Convolutional Attention block"""
    # Temporal Convolution with dilation
    conv_out = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        name=f'tca_conv_{block_id}'
    )(inputs)
    
    # Layer normalization
    conv_out = LayerNormalization(name=f'tca_norm1_{block_id}')(conv_out)
    
    # Multi-head attention
    attention_out = MultiHeadAttention(
        num_heads=4,
        key_dim=filters // 4,
        name=f'tca_attention_{block_id}'
    )(conv_out, conv_out)
    
    # Add & Norm
    attention_out = Add(name=f'tca_add1_{block_id}')([conv_out, attention_out])
    attention_out = LayerNormalization(name=f'tca_norm2_{block_id}')(attention_out)
    
    # Feed-forward
    ff_out = Dense(filters * 2, activation='relu', name=f'tca_ff1_{block_id}')(attention_out)
    ff_out = Dropout(dropout_rate, name=f'tca_dropout1_{block_id}')(ff_out)
    ff_out = Dense(filters, name=f'tca_ff2_{block_id}')(ff_out)
    ff_out = Dropout(dropout_rate, name=f'tca_dropout2_{block_id}')(ff_out)
    
    # Add & Norm
    output = Add(name=f'tca_add2_{block_id}')([attention_out, ff_out])
    output = LayerNormalization(name=f'tca_norm3_{block_id}')(output)
    
    return output

def build_tca_transformer_model(input_shape):
    """Build TCA (Temporal Convolutional Attention) Transformer model"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Input
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # Initial projection
        x = Dense(TRANSFORMER_D_MODEL, name='input_projection')(inputs)
        
        # Positional encoding
        pos_encoding = positional_encoding(input_shape[0], TRANSFORMER_D_MODEL)
        x = x + pos_encoding
        
        # TCA blocks with different dilation rates
        for i, (filters, dilation) in enumerate(zip(TCA_FILTERS, TCA_DILATION_RATES)):
            x = create_tca_block(
                x, filters, TCA_KERNEL_SIZE, dilation, 
                TRANSFORMER_DROPOUT, f'block_{i}'
            )
        
        # Global attention mechanism
        global_attention = MultiHeadAttention(
            num_heads=TRANSFORMER_NUM_HEADS,
            key_dim=TRANSFORMER_D_MODEL // TRANSFORMER_NUM_HEADS,
            name='global_attention'
        )(x, x)
        
        x = Add(name='global_add')([x, global_attention])
        x = LayerNormalization(name='global_norm')(x)
        
        # Temporal pooling
        pooled = GlobalAveragePooling1D(name='temporal_pooling')(x)
        
        # Final prediction layers
        dense1 = Dense(256, activation='relu', name='dense1')(pooled)
        dense1 = Dropout(TRANSFORMER_DROPOUT, name='dropout1')(dense1)
        
        dense2 = Dense(128, activation='relu', name='dense2')(dense1)
        dense2 = Dropout(TRANSFORMER_DROPOUT, name='dropout2')(dense2)
        
        dense3 = Dense(64, activation='relu', name='dense3')(dense2)
        output = Dense(1, activation='linear', name='output', dtype='float32')(dense3)
        
        model = Model(inputs=inputs, outputs=output, name='tca_transformer')
        
        # Compile with custom learning rate
        optimizer = Adam(learning_rate=TRANSFORMER_LEARNING_RATE, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"[TCA-TRANSFORMER] Model created with input shape: {input_shape}")
        print(f"[TCA-TRANSFORMER] Model parameters: {model.count_params():,}")
        return model

def build_standard_transformer_model(input_shape):
    """Build standard Transformer model for comparison"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # Input projection
        x = Dense(TRANSFORMER_D_MODEL, name='input_projection')(inputs)
        
        # Positional encoding
        pos_encoding = positional_encoding(input_shape[0], TRANSFORMER_D_MODEL)
        x = x + pos_encoding
        
        # Standard transformer layers
        for i in range(TRANSFORMER_NUM_LAYERS):
            # Multi-head attention
            attention_out = MultiHeadAttention(
                num_heads=TRANSFORMER_NUM_HEADS,
                key_dim=TRANSFORMER_D_MODEL // TRANSFORMER_NUM_HEADS,
                name=f'attention_{i}'
            )(x, x)
            
            # Add & Norm
            x = Add(name=f'add1_{i}')([x, attention_out])
            x = LayerNormalization(name=f'norm1_{i}')(x)
            
            # Feed-forward network
            ff_out = Dense(TRANSFORMER_DFF, activation='relu', name=f'ff1_{i}')(x)
            ff_out = Dropout(TRANSFORMER_DROPOUT, name=f'dropout1_{i}')(ff_out)
            ff_out = Dense(TRANSFORMER_D_MODEL, name=f'ff2_{i}')(ff_out)
            ff_out = Dropout(TRANSFORMER_DROPOUT, name=f'dropout2_{i}')(ff_out)
            
            # Add & Norm
            x = Add(name=f'add2_{i}')([x, ff_out])
            x = LayerNormalization(name=f'norm2_{i}')(x)
        
        # Global pooling and output
        pooled = GlobalAveragePooling1D(name='global_pooling')(x)
        dense = Dense(128, activation='relu', name='dense')(pooled)
        dense = Dropout(TRANSFORMER_DROPOUT, name='final_dropout')(dense)
        output = Dense(1, activation='linear', name='output', dtype='float32')(dense)
        
        model = Model(inputs=inputs, outputs=output, name='standard_transformer')
        
        optimizer = Adam(learning_rate=TRANSFORMER_LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"[STANDARD-TRANSFORMER] Model created with input shape: {input_shape}")
        print(f"[STANDARD-TRANSFORMER] Model parameters: {model.count_params():,}")
        return model

def tca_transformer_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """TCA Transformer forecast"""
    print("[TCA-TRANSFORMER] Starting TCA Transformer forecasting...")
    
    if len(demand_series) < TRANSFORMER_LOOKBACK + 200:
        print(f"[TCA-TRANSFORMER] Insufficient data. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        X, y, scaler, feature_cols = prepare_transformer_data(demand_series, weekend_series, weather_df, TRANSFORMER_LOOKBACK)
        
        if len(X) < 100:
            print(f"[TCA-TRANSFORMER] Too few training samples. Using seasonal naive fallback.")
            return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"[TCA-TRANSFORMER] Training on {len(X_train):,} samples")
        print(f"[TCA-TRANSFORMER] Input shape: {X_train.shape}")
        
        # Build and train model
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = build_tca_transformer_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
            ]
            
            print("[TCA-TRANSFORMER] Starting GPU training...")
            start_time = datetime.now()
            
            history = model.fit(
                X_train, y_train,
                epochs=TRANSFORMER_EPOCHS,
                batch_size=TRANSFORMER_BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = datetime.now() - start_time
            print(f"[TCA-TRANSFORMER] Training completed in {training_time.total_seconds():.2f} seconds")
        
        # Generate forecasts
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            forecasts = []
            
            for step in range(steps):
                pred = model.predict(last_sequence, verbose=0)[0, 0]
                forecasts.append(pred)
                
                # Update sequence
                new_row = last_sequence[0, -1].copy()
                new_row[0] = pred
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1] = new_row
        
        # Inverse transform
        dummy_features = np.zeros((len(forecasts), len(feature_cols)))
        dummy_features[:, 0] = forecasts
        forecasts_scaled = scaler.inverse_transform(dummy_features)[:, 0]
        
        # Create forecast series
        last_stamp = demand_series.index[-1]
        future_index = pd.date_range(start=last_stamp + pd.Timedelta(FREQ), periods=steps, freq=FREQ)
        
        print(f"[TCA-TRANSFORMER] Successfully generated {len(forecasts_scaled)} forecasts")
        return pd.Series(forecasts_scaled, index=future_index, name="tca_transformer")
        
    except Exception as e:
        print(f"[TCA-TRANSFORMER] Error: {e}. Using seasonal naive fallback.")
        import traceback
        traceback.print_exc()
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)

def standard_transformer_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """Standard Transformer forecast for comparison"""
    print("[STANDARD-TRANSFORMER] Starting Standard Transformer forecasting...")
    
    if len(demand_series) < TRANSFORMER_LOOKBACK + 200:
        print(f"[STANDARD-TRANSFORMER] Insufficient data. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        X, y, scaler, feature_cols = prepare_transformer_data(demand_series, weekend_series, weather_df, TRANSFORMER_LOOKBACK)
        
        if len(X) < 100:
            print(f"[STANDARD-TRANSFORMER] Too few training samples. Using seasonal naive fallback.")
            return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
        
        # Split data
        train_size = int(len(X) * 0.8)# filepath: c:\Users\user\OneDrive\Documents\!DemandUROP\Processing\ensemble_forecast_transformer.py
# pip install pandas numpy matplotlib openpyxl requests tensorflow scikit-learn

# GPU Memory Optimization and Setup
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
for g in tf.config.list_physical_devices('GPU'):
    try: 
        tf.config.experimental.set_memory_growth(g, True)
        tf.config.experimental.set_virtual_device_configuration(
            g, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6144)]
        )
    except: pass

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports for Transformer
try:
    import keras
    from keras.models import Model
    from keras.layers import (
        Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
        GlobalAveragePooling1D, Conv1D, Add, Embedding, Concatenate
    )
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    print(f"[tensorflow] Using TensorFlow {tf.__version__} with Keras {keras.__version__}")
except ImportError:
    try:
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import (
            Input, Dense, Dropout, LayerNormalization, MultiHeadAttention,
            GlobalAveragePooling1D, Conv1D, Add, Embedding, Concatenate
        )
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
        print(f"[tensorflow] Using TensorFlow {tf.__version__} with built-in Keras")
    except ImportError as e:
        print(f"[error] Cannot import Keras: {e}")
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
K_DAYS_AVG = 7

# Weather variables to fetch
WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m", 
    "precipitation",
    "surface_pressure",
    "wind_speed_10m"
]

# Transformer parameters
TRANSFORMER_LOOKBACK = 7 * PERIOD  # 7 days of history
TRANSFORMER_EPOCHS = 100
TRANSFORMER_BATCH_SIZE = 64  # Smaller batch for transformer
TRANSFORMER_D_MODEL = 128  # Model dimension
TRANSFORMER_NUM_HEADS = 8  # Attention heads
TRANSFORMER_NUM_LAYERS = 4  # Transformer layers
TRANSFORMER_DFF = 512  # Feed-forward dimension
TRANSFORMER_DROPOUT = 0.1
TRANSFORMER_LEARNING_RATE = 1e-4

# TCA (Temporal Convolutional Attention) parameters
TCA_FILTERS = [64, 128, 256]  # Convolutional filters
TCA_KERNEL_SIZE = 3
TCA_DILATION_RATES = [1, 2, 4, 8]  # Dilated convolutions for temporal patterns
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
    lat = match.iloc[0]['X_EQ']
    lon = match.iloc[0]['Y_EQ']
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

def prepare_transformer_data(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, lookback: int):
    """Prepare data for Transformer training with temporal features"""
    print(f"[TRANSFORMER] Using enhanced features: demand, temperature_2m, is_weekend + temporal features")
    
    # Limit to recent data for faster training
    cutoff_date = demand_series.index[-1] - pd.Timedelta(days=120)  # 4 months
    recent_demand = demand_series.loc[cutoff_date:]
    recent_weekend = weekend_series.loc[cutoff_date:]
    
    print(f"[TRANSFORMER] Limited to recent {len(recent_demand)} points (last 120 days)")
    
    # Create base features
    features_dict = {'demand': recent_demand}
    features_dict['is_weekend'] = recent_weekend.astype(float)
    
    # Add weather features
    if not weather_df.empty and 'temperature_2m' in weather_df.columns:
        recent_weather = weather_df.loc[cutoff_date:]
        common_index = recent_demand.index.intersection(recent_weather.index)
        if len(common_index) > 0:
            temp_aligned = recent_weather.loc[common_index, 'temperature_2m']
            features_dict['temperature_2m'] = temp_aligned
            
            # Add more weather features for transformer
            if 'relative_humidity_2m' in recent_weather.columns:
                humidity_aligned = recent_weather.loc[common_index, 'relative_humidity_2m']
                features_dict['humidity'] = humidity_aligned
            
        print("[TRANSFORMER] Added weather features")
    else:
        features_dict['temperature_2m'] = pd.Series(0.0, index=recent_demand.index)
        features_dict['humidity'] = pd.Series(0.0, index=recent_demand.index)
    
    # Add temporal features for transformer
    timestamps = recent_demand.index
    features_dict['hour_of_day'] = [t.hour + t.minute/60.0 for t in timestamps]
    features_dict['day_of_week'] = [float(t.weekday()) for t in timestamps]
    features_dict['day_of_year'] = [float(t.dayofyear) for t in timestamps]
    
    # Add cyclical features
    features_dict['hour_sin'] = np.sin(2 * np.pi * np.array(features_dict['hour_of_day']) / 24)
    features_dict['hour_cos'] = np.cos(2 * np.pi * np.array(features_dict['hour_of_day']) / 24)
    features_dict['day_sin'] = np.sin(2 * np.pi * np.array(features_dict['day_of_week']) / 7)
    features_dict['day_cos'] = np.cos(2 * np.pi * np.array(features_dict['day_of_week']) / 7)
    
    # Combine features
    features = pd.DataFrame(features_dict)
    features = features.dropna()
    
    print(f"[TRANSFORMER] Feature matrix shape: {features.shape}")
    print(f"[TRANSFORMER] Features: {list(features.columns)}")
    
    if len(features) < lookback + 100:
        raise ValueError(f"Insufficient data: {len(features)} points, need at least {lookback + 100}")
    
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Create sequences with limited data for speed
    max_sequences = min(3000, len(scaled_features) - lookback)
    X, y = [], []
    
    start_idx = max(lookback, len(scaled_features) - max_sequences)
    for i in range(start_idx, len(scaled_features)):
        X.append(scaled_features[i-lookback:i])
        y.append(scaled_features[i, 0])  # demand is first column
    
    print(f"[TRANSFORMER] Created {len(X)} sequences")
    return np.array(X), np.array(y), scaler, features.columns

def positional_encoding(length, d_model):
    """Create positional encoding for transformer"""
    angle_rads = np.arange(length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

def create_tca_block(inputs, filters, kernel_size, dilation_rate, dropout_rate, block_id):
    """Create a Temporal Convolutional Attention block"""
    # Temporal Convolution with dilation
    conv_out = Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding='causal',
        activation='relu',
        name=f'tca_conv_{block_id}'
    )(inputs)
    
    # Layer normalization
    conv_out = LayerNormalization(name=f'tca_norm1_{block_id}')(conv_out)
    
    # Multi-head attention
    attention_out = MultiHeadAttention(
        num_heads=4,
        key_dim=filters // 4,
        name=f'tca_attention_{block_id}'
    )(conv_out, conv_out)
    
    # Add & Norm
    attention_out = Add(name=f'tca_add1_{block_id}')([conv_out, attention_out])
    attention_out = LayerNormalization(name=f'tca_norm2_{block_id}')(attention_out)
    
    # Feed-forward
    ff_out = Dense(filters * 2, activation='relu', name=f'tca_ff1_{block_id}')(attention_out)
    ff_out = Dropout(dropout_rate, name=f'tca_dropout1_{block_id}')(ff_out)
    ff_out = Dense(filters, name=f'tca_ff2_{block_id}')(ff_out)
    ff_out = Dropout(dropout_rate, name=f'tca_dropout2_{block_id}')(ff_out)
    
    # Add & Norm
    output = Add(name=f'tca_add2_{block_id}')([attention_out, ff_out])
    output = LayerNormalization(name=f'tca_norm3_{block_id}')(output)
    
    return output

def build_tca_transformer_model(input_shape):
    """Build TCA (Temporal Convolutional Attention) Transformer model"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        # Input
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # Initial projection
        x = Dense(TRANSFORMER_D_MODEL, name='input_projection')(inputs)
        
        # Positional encoding
        pos_encoding = positional_encoding(input_shape[0], TRANSFORMER_D_MODEL)
        x = x + pos_encoding
        
        # TCA blocks with different dilation rates
        for i, (filters, dilation) in enumerate(zip(TCA_FILTERS, TCA_DILATION_RATES)):
            x = create_tca_block(
                x, filters, TCA_KERNEL_SIZE, dilation, 
                TRANSFORMER_DROPOUT, f'block_{i}'
            )
        
        # Global attention mechanism
        global_attention = MultiHeadAttention(
            num_heads=TRANSFORMER_NUM_HEADS,
            key_dim=TRANSFORMER_D_MODEL // TRANSFORMER_NUM_HEADS,
            name='global_attention'
        )(x, x)
        
        x = Add(name='global_add')([x, global_attention])
        x = LayerNormalization(name='global_norm')(x)
        
        # Temporal pooling
        pooled = GlobalAveragePooling1D(name='temporal_pooling')(x)
        
        # Final prediction layers
        dense1 = Dense(256, activation='relu', name='dense1')(pooled)
        dense1 = Dropout(TRANSFORMER_DROPOUT, name='dropout1')(dense1)
        
        dense2 = Dense(128, activation='relu', name='dense2')(dense1)
        dense2 = Dropout(TRANSFORMER_DROPOUT, name='dropout2')(dense2)
        
        dense3 = Dense(64, activation='relu', name='dense3')(dense2)
        output = Dense(1, activation='linear', name='output', dtype='float32')(dense3)
        
        model = Model(inputs=inputs, outputs=output, name='tca_transformer')
        
        # Compile with custom learning rate
        optimizer = Adam(learning_rate=TRANSFORMER_LEARNING_RATE, clipnorm=1.0)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        print(f"[TCA-TRANSFORMER] Model created with input shape: {input_shape}")
        print(f"[TCA-TRANSFORMER] Model parameters: {model.count_params():,}")
        return model

def build_standard_transformer_model(input_shape):
    """Build standard Transformer model for comparison"""
    with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
        inputs = Input(shape=input_shape, name='input_sequence')
        
        # Input projection
        x = Dense(TRANSFORMER_D_MODEL, name='input_projection')(inputs)
        
        # Positional encoding
        pos_encoding = positional_encoding(input_shape[0], TRANSFORMER_D_MODEL)
        x = x + pos_encoding
        
        # Standard transformer layers
        for i in range(TRANSFORMER_NUM_LAYERS):
            # Multi-head attention
            attention_out = MultiHeadAttention(
                num_heads=TRANSFORMER_NUM_HEADS,
                key_dim=TRANSFORMER_D_MODEL // TRANSFORMER_NUM_HEADS,
                name=f'attention_{i}'
            )(x, x)
            
            # Add & Norm
            x = Add(name=f'add1_{i}')([x, attention_out])
            x = LayerNormalization(name=f'norm1_{i}')(x)
            
            # Feed-forward network
            ff_out = Dense(TRANSFORMER_DFF, activation='relu', name=f'ff1_{i}')(x)
            ff_out = Dropout(TRANSFORMER_DROPOUT, name=f'dropout1_{i}')(ff_out)
            ff_out = Dense(TRANSFORMER_D_MODEL, name=f'ff2_{i}')(ff_out)
            ff_out = Dropout(TRANSFORMER_DROPOUT, name=f'dropout2_{i}')(ff_out)
            
            # Add & Norm
            x = Add(name=f'add2_{i}')([x, ff_out])
            x = LayerNormalization(name=f'norm2_{i}')(x)
        
        # Global pooling and output
        pooled = GlobalAveragePooling1D(name='global_pooling')(x)
        dense = Dense(128, activation='relu', name='dense')(pooled)
        dense = Dropout(TRANSFORMER_DROPOUT, name='final_dropout')(dense)
        output = Dense(1, activation='linear', name='output', dtype='float32')(dense)
        
        model = Model(inputs=inputs, outputs=output, name='standard_transformer')
        
        optimizer = Adam(learning_rate=TRANSFORMER_LEARNING_RATE, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        print(f"[STANDARD-TRANSFORMER] Model created with input shape: {input_shape}")
        print(f"[STANDARD-TRANSFORMER] Model parameters: {model.count_params():,}")
        return model

def tca_transformer_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """TCA Transformer forecast"""
    print("[TCA-TRANSFORMER] Starting TCA Transformer forecasting...")
    
    if len(demand_series) < TRANSFORMER_LOOKBACK + 200:
        print(f"[TCA-TRANSFORMER] Insufficient data. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        X, y, scaler, feature_cols = prepare_transformer_data(demand_series, weekend_series, weather_df, TRANSFORMER_LOOKBACK)
        
        if len(X) < 100:
            print(f"[TCA-TRANSFORMER] Too few training samples. Using seasonal naive fallback.")
            return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        print(f"[TCA-TRANSFORMER] Training on {len(X_train):,} samples")
        print(f"[TCA-TRANSFORMER] Input shape: {X_train.shape}")
        
        # Build and train model
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            model = build_tca_transformer_model((X_train.shape[1], X_train.shape[2]))
            
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1)
            ]
            
            print("[TCA-TRANSFORMER] Starting GPU training...")
            start_time = datetime.now()
            
            history = model.fit(
                X_train, y_train,
                epochs=TRANSFORMER_EPOCHS,
                batch_size=TRANSFORMER_BATCH_SIZE,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            training_time = datetime.now() - start_time
            print(f"[TCA-TRANSFORMER] Training completed in {training_time.total_seconds():.2f} seconds")
        
        # Generate forecasts
        with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
            last_sequence = X[-1].reshape(1, X.shape[1], X.shape[2])
            forecasts = []
            
            for step in range(steps):
                pred = model.predict(last_sequence, verbose=0)[0, 0]
                forecasts.append(pred)
                
                # Update sequence
                new_row = last_sequence[0, -1].copy()
                new_row[0] = pred
                last_sequence = np.roll(last_sequence, -1, axis=1)
                last_sequence[0, -1] = new_row
        
        # Inverse transform
        dummy_features = np.zeros((len(forecasts), len(feature_cols)))
        dummy_features[:, 0] = forecasts
        forecasts_scaled = scaler.inverse_transform(dummy_features)[:, 0]
        
        # Create forecast series
        last_stamp = demand_series.index[-1]
        future_index = pd.date_range(start=last_stamp + pd.Timedelta(FREQ), periods=steps, freq=FREQ)
        
        print(f"[TCA-TRANSFORMER] Successfully generated {len(forecasts_scaled)} forecasts")
        return pd.Series(forecasts_scaled, index=future_index, name="tca_transformer")
        
    except Exception as e:
        print(f"[TCA-TRANSFORMER] Error: {e}. Using seasonal naive fallback.")
        import traceback
        traceback.print_exc()
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)

def standard_transformer_forecast(demand_series: pd.Series, weekend_series: pd.Series, weather_df: pd.DataFrame, steps: int) -> pd.Series:
    """Standard Transformer forecast for comparison"""
    print("[STANDARD-TRANSFORMER] Starting Standard Transformer forecasting...")
    
    if len(demand_series) < TRANSFORMER_LOOKBACK + 200:
        print(f"[STANDARD-TRANSFORMER] Insufficient data. Using seasonal naive fallback.")
        return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
    
    try:
        X, y, scaler, feature_cols = prepare_transformer_data(demand_series, weekend_series, weather_df, TRANSFORMER_LOOKBACK)
        
        if len(X) < 100:
            print(f"[STANDARD-TRANSFORMER] Too few training samples. Using seasonal naive fallback.")
            return seasonal_naive_forecast(demand_series, steps, PERIOD, K_DAYS_AVG)
        
        #