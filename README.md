## Features

### Core Functionality
- **Multi-Model Ensemble**: Combines Seasonal Naive, LSTM, and N-BEATS forecasting models
- **Real-time Weather Integration**: Fetches weather data from Open-Meteo API with automatic alignment
- **GPU Acceleration**: Optimized for NVIDIA GPUs with mixed precision training
- **Automated Data Processing**: Handles missing values, duplicates, and time series regularization
- **Weekend/Weekday Analysis**: Incorporates weekend patterns for improved accuracy

### Model Architectures
1. **Seasonal Naive Baseline**: K-day averaging with seasonal decomposition
2. **LSTM Ensemble**: Multi-layer LSTM with dropout and recurrent dropout
3. **N-BEATS (Darts)**: Professional N-BEATS implementation using PyTorch backend

### Data Sources
- **Historical Demand**: Excel files with substation-level consumption data
- **Weather Data**: Temperature, humidity, precipitation, pressure, wind speed
- **Temporal Features**: Weekend/weekday indicators, time-based patterns
- **Geospatial**: Substation coordinates for weather data alignment

## Project Structure

```
DemandUROP/
├── Data/
│   └── demand_data.xlsx          # Main dataset with demand history
├── Processing/
│   ├── ensemble_forecast_nbeats.py    # Main forecasting script
│   ├── add_weekday_columns.py         # Weekend feature engineering
│   └── test_gpu.py                    # GPU compatibility testing
├── Output/
│   ├── forecast_comparison_*.png       # Visualization outputs
│   ├── forecast_results_*.csv         # Numerical results
│   └── metrics_*.txt                  # Performance metrics
└── README.md
```

### Dependencies
```bash
pip install pandas numpy matplotlib openpyxl requests tensorflow scikit-learn darts[torch]
```



### 1. Prepare Data
Ensure your Excel file contains:
- `DATE/HOUR` column with timestamps
- Substation demand columns
- `is_weekend` boolean column (auto-generated if missing)

### 2. Add Weekend Features (if needed)
```bash
python Processing/add_weekday_columns.py
```

### 3. Run Forecasting
```bash
cd Processing
python ensemble_forecast_nbeats.py
```

### 4. Enter Substation Name
When prompted, enter the exact substation name as it appears in your data.

## 📊 Model Performance

### LSTM Configuration
- **Architecture**: 128→128→64 LSTM layers with dropout
- **Features**: Demand history, temperature, weekend indicators
- **Lookback**: 3 days (144 timesteps)
- **Training**: GPU-accelerated with early stopping

### N-BEATS Configuration
- **Stacks**: 30 trend + seasonality stacks
- **Layers**: 4 layers with 512 units each
- **Backend**: PyTorch with CUDA acceleration
- **Architecture**: Generic N-BEATS for flexibility

### Performance Metrics
- **MAPE**: Mean Absolute Percentage Error
- **MSE**: Mean Squared Error  
- **wMAPE**: Weighted Mean Absolute Percentage Error

## 🔧 Configuration

### Key Parameters (in `ensemble_forecast_nbeats.py`)
```python
# Time settings
FREQ = "30min"              # Data frequency
FORECAST_DAYS = 2           # Forecast horizon
PERIOD = 48                 # Periods per day

# LSTM settings
LSTM_LOOKBACK = 3 * PERIOD  # 3 days history
LSTM_EPOCHS = 20            # Training epochs
LSTM_BATCH_SIZE = 512       # Batch size for GPU

# N-BEATS settings
NBEATS_INPUT_CHUNK_LENGTH = 7 * PERIOD   # 7 days input
NBEATS_OUTPUT_CHUNK_LENGTH = 2 * PERIOD  # 2 days output
NBEATS_EPOCHS = 100         # Training epochs
```

## Usage Examples

### Basic Forecasting
```python
from ensemble_forecast_nbeats import main
main()  # Interactive mode
```

### Programmatic Usage
```python
# Load data
demand_series, weekend_series = load_and_regularise(EXCEL_PATH, "SUBSTATION_NAME")

# Generate forecasts
lstm_forecast = lstm_ensemble_forecast(demand_series, weekend_series, weather_df, 96)
nbeats_forecast = nbeats_darts_forecast(demand_series, weekend_series, weather_df, 96)
```

## 📈 Output Files

### Generated Files
- `forecast_comparison_darts_[SUBSTATION].png`: Visual comparison of all models
- `forecast_results_darts_[SUBSTATION].csv`: Numerical forecasts and historical data
- `metrics_darts_[SUBSTATION].txt`: Performance comparison table

### Sample Output
```
FORECAST PERFORMANCE COMPARISON - SUBSTATION_NAME
===============================================
Metric       Seasonal Naive  LSTM Ensemble   N-BEATS Darts   Best Model
---------------------------------------------------------------------------
MAPE         7.608           5.234           4.891           N-BEATS
MSE          0.082           0.045           0.038           N-BEATS
wMAPE        7.670           5.123           4.756           N-BEATS
```

## ⚡ Performance Optimization

### GPU Utilization
- **Memory Management**: Automatic GPU memory growth
- **Mixed Precision**: Float16 training for speed
- **Batch Optimization**: Large batches for GPU efficiency
- **XLA Compilation**: Accelerated linear algebra

### Speed Optimizations
- **Data Limiting**: 90-day training window for LSTM
- **Sequence Capping**: Maximum 2000 training sequences
- **Early Stopping**: Aggressive patience settings
- **Parallel Processing**: Multi-threaded data loading


#### Memory Errors
- Reduce `LSTM_BATCH_SIZE` or `NBEATS_BATCH_SIZE`
- Decrease lookback windows
- Use CPU-only mode by setting `CUDA_VISIBLE_DEVICES=""`

#### Data Loading Issues
- Verify Excel file structure matches expected format
- Check for missing `is_weekend` column
- Ensure date formats are consistent

## Technical Details

### Weather Data Integration
- **Source**: Open-Meteo Historical Weather API
- **Preprocessing**: Interpolation and forward-filling for missing values

### Feature Engineering
- **Temporal**: Weekend/weekday binary indicators
- **Seasonal**: Day-of-week patterns and seasonal decomposition
- **Weather**: Temperature integration with lag analysis
- **Scaling**: MinMax normalization for neural networks

### Model Architecture Details

#### LSTM Ensemble
```
Input(samples, timesteps=144, features=3)
├── LSTM(128, return_sequences=True, dropout=0.2)
├── LSTM(128, return_sequences=True, dropout=0.2) 
├── LSTM(64, return_sequences=False, dropout=0.2)
├── Dense(64, activation='relu')
├── Dropout(0.3)
├── Dense(32, activation='relu')
├── Dropout(0.2)
└── Dense(1, activation='linear')
```

#### N-BEATS Architecture
- **Interpretable**: Trend and seasonality decomposition
- **Generic**: Flexible architecture for various patterns
- **Residual**: Hierarchical residual connections
- **Attention**: Built-in feature importance weighting

### Libraries Used
- **Darts**: Professional time series forecasting library
- **TensorFlow**: Deep learning framework for LSTM
- **PyTorch**: Backend for N-BEATS implementation
- **Pandas**: Data manipulation and time series handling




## 📄 License



### Planned Features
- [ ] Transformer-based models (Attention mechanisms)
- [ ] Multi-horizon forecasting (1-7 days)
- [ ] Uncertainty quantification

- [ ] Model compression for deployment

---

**Last Updated**: August 2025