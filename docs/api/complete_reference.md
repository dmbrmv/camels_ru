# API Reference

Complete API documentation for the CAMELS-RU project modules.

## Module Structure

```
src/
├── hydro_metrics.py          # Core hydrological calculations
├── processing.py             # Data processing utilities
├── hydro/                    # Hydrological analysis package
│   ├── base_flow.py         # Base flow separation methods
│   ├── flow_duration.py     # Flow duration curve analysis
│   ├── flow_extremes.py     # Extreme event detection
│   ├── flow_indices.py      # Flow characteristic indices
│   ├── flow_timing.py       # Temporal flow characteristics
│   └── flow_variability.py  # Flow variability measures
├── meteo/                    # Meteorological analysis package
│   ├── temperature.py       # Temperature analysis tools
│   ├── climate_indices.py   # Drought and climate indices
│   ├── extremes.py         # Extreme weather events
│   └── seasonal_stats.py   # Seasonal statistics
├── timeseries_stats/        # Time series analysis tools
│   ├── trends.py           # Trend analysis
│   └── statistics.py       # Statistical measures
└── utils/                   # Utility functions
    └── logger.py           # Logging configuration
```

## Core Modules

### hydro_metrics.py

Main module for hydrological metric calculations.

#### Functions

##### `FirstPass(Q: np.ndarray, alpha: float) -> list[list[float]]`

Performs the first pass of the Eckhardt baseflow separation filter.

**Parameters:**

- `Q`: Discharge time series as numpy array
- `alpha`: Filter parameter (0.9-0.98)

**Returns:**

- List containing [quick_flow, base_flow] components

**Performance:**

- JIT-compiled with Numba for optimal speed
- Memory complexity: O(n) where n is series length

---

##### `BackwardPass(Q_forward_1: list[list[float]], alpha: float) -> list[list[float]]`

Performs the backward pass refinement of baseflow separation.

**Parameters:**

- `Q_forward_1`: Output from FirstPass function
- `alpha`: Filter parameter (same as first pass)

**Returns:**

- Refined [quick_flow, base_flow] components

---

##### `ForwardPass(Q_backward: list[list[float]], alpha: float) -> list[list[float]]`

Performs the final forward pass for smooth baseflow separation.

**Parameters:**

- `Q_backward`: Output from BackwardPass function
- `alpha`: Filter parameter (same as previous passes)

**Returns:**

- Final [quick_flow, base_flow] components

---

##### `bfi(Q: np.ndarray, alpha: float, passes: int, reflect: int) -> tuple[float, np.ndarray]`

Calculates Base Flow Index using specified parameters.

**Parameters:**

- `Q`: Discharge time series
- `alpha`: Filter parameter (0.9-0.98)
- `passes`: Number of filtering passes (typically 3)
- `reflect`: Edge reflection length (typically 30)

**Returns:**

- Tuple of (BFI_value, baseflow_series)

**Example:**

```python
import numpy as np
from src.hydro_metrics import bfi

discharge = np.array([1.2, 1.5, 2.1, 1.8, 1.3, 1.1, 0.9])
bfi_value, baseflow = bfi(discharge, alpha=0.925, passes=3, reflect=5)
print(f"BFI: {bfi_value:.3f}")
```

---

##### `bfi_1000(Q: np.ndarray, passes: int, reflect: int) -> tuple[float, np.ndarray]`

Calculates BFI with uncertainty estimation using 1000 random alpha values.

**Parameters:**

- `Q`: Discharge time series
- `passes`: Number of filtering passes
- `reflect`: Edge reflection length

**Returns:**

- Tuple of (mean_BFI, mean_baseflow_series)

**Algorithm:**

1. Generate 1000 random α values between 0.9 and 0.98
2. Calculate BFI for each α
3. Return mean values across all realizations

**Reproducibility:**

- Uses fixed random seed (1996) for consistent results
- All α values are pre-generated before processing

---

##### `slope_fdc_gauge(hydro_year: pd.Series) -> float`

Calculates the slope of the flow duration curve.

**Parameters:**

- `hydro_year`: Daily discharge data for one hydrological year

**Returns:**

- Flow duration curve slope value

**Formula:**

```
slope = (log(Q33) - log(Q67)) / (0.67 - 0.33)
```

Where Q33 and Q67 are the 33rd and 67th percentiles.

---

##### `hfd_calc(calendar_year: pd.Series, hydro_year: pd.Series) -> pd.Timestamp`

Calculates the half flow date (center of mass of annual hydrograph).

**Parameters:**

- `calendar_year`: Calendar year discharge data
- `hydro_year`: Hydrological year discharge data

**Returns:**

- Date when cumulative discharge reaches 50% of annual total

**Applications:**

- Snowmelt timing assessment
- Climate change impact studies
- Regional flow regime classification

---

##### `q5_q95(hydro_year: pd.Series) -> dict[str, float]`

Calculates high and low flow quantiles.

**Parameters:**

- `hydro_year`: Daily discharge data

**Returns:**

- Dictionary with keys 'q5' (high flow) and 'q95' (low flow)

**Usage:**

```python
quantiles = q5_q95(discharge_data)
print(f"High flow (Q5): {quantiles['q5']:.2f} m³/s")
print(f"Low flow (Q95): {quantiles['q95']:.2f} m³/s")
```

---

##### `high_q_freq(hydro_year: pd.Series) -> pd.Series`

Identifies high flow events (> 9 × median daily flow).

**Parameters:**

- `hydro_year`: Daily discharge data

**Returns:**

- Series containing only high flow values (others as NaN)

---

##### `low_q_freq(hydro_year: pd.Series) -> pd.Series`

Identifies low flow events (< 0.2 × mean daily flow).

**Parameters:**

- `hydro_year`: Daily discharge data

**Returns:**

- Series containing only low flow values (others as NaN)

---

##### `high_q_dur(hydro_year: pd.Series) -> list[pd.Series]`

Identifies continuous high flow periods.

**Parameters:**

- `hydro_year`: Daily discharge data

**Returns:**

- List of Series, each representing a continuous high flow event

**Threshold:** 2 × median daily flow

---

##### `low_q_dur(hydro_year: pd.Series) -> list[pd.Series]`

Identifies continuous low flow periods.

**Parameters:**

- `hydro_year`: Daily discharge data

**Returns:**

- List of Series, each representing a continuous low flow event

**Threshold:** Below mean daily flow

---

##### `hydro_job(hydro_year: pd.Series, calendar_year: pd.Series) -> None`

Comprehensive hydrological analysis with visualization.

**Parameters:**

- `hydro_year`: Hydrological year discharge data
- `calendar_year`: Calendar year discharge data

**Returns:**

- None (generates plots)

**Outputs:**

- Multi-panel plot showing:
  - Daily discharge hydrograph
  - Baseflow separation
  - Quantile thresholds
  - Half flow date
  - High/low flow periods

**Example:**

```python
from src.hydro_metrics import hydro_job
from src.processing import split_by_hydro_year

# Split data by hydrological year
hydro_years = split_by_hydro_year(discharge_data)

# Analyze each year
for year, data in hydro_years.items():
    if len(data) > 300:  # Ensure sufficient data
        hydro_job(data, data)  # Use same data for both parameters
```

### processing.py

Data processing utilities for time series manipulation.

#### Functions

##### `split_by_hydro_year(discharge_obs: pd.Series) -> dict[str, pd.Series]`

Splits discharge time series by hydrological year (Oct 1 - Sep 30).

**Parameters:**

- `discharge_obs`: Pandas Series with DatetimeIndex

**Returns:**

- Dictionary with year strings as keys and yearly data as values

**Error Handling:**

- Raises ValueError for empty input series
- Logs errors for debugging

**Example:**

```python
from src.processing import split_by_hydro_year

hydro_years = split_by_hydro_year(discharge_data)
for year, data in hydro_years.items():
    print(f"Year {year}: {len(data)} observations")
```

---

##### `split_by_year(discharge_obs: pd.Series) -> dict[str, pd.Series]`

Splits discharge time series by calendar year (Jan 1 - Dec 31).

**Parameters:**

- `discharge_obs`: Pandas Series with DatetimeIndex

**Returns:**

- Dictionary with year strings as keys and yearly data as values

**Use Cases:**

- Calendar year statistics
- Annual reporting
- Cross-year comparison studies

## Meteorological Analysis Package

### meteo/temperature.py

#### Class: TemperatureAnalysis

Comprehensive temperature analysis tools.

##### `__init__(self)`

Initialize TemperatureAnalysis instance.

##### `basic_statistics(self, temperature: pd.Series) -> dict`

Calculate comprehensive temperature statistics.

**Parameters:**

- `temperature`: Daily temperature data

**Returns:**

- Dictionary containing:
  - Central tendency: mean, median, mode
  - Variability: std, variance, IQR, CV
  - Distribution: skewness, kurtosis
  - Extremes: min, max, percentiles

**Example:**

```python
from src.meteo.temperature import TemperatureAnalysis

analyzer = TemperatureAnalysis()
stats = analyzer.basic_statistics(temperature_data)
print(f"Mean temperature: {stats['mean']:.1f}°C")
print(f"Temperature range: {stats['range']:.1f}°C")
```

##### `annual_statistics(self, temperature: pd.Series) -> pd.DataFrame`

Calculate annual temperature statistics.

**Parameters:**

- `temperature`: Multi-year daily temperature data

**Returns:**

- DataFrame with annual statistics by year

##### `seasonal_statistics(self, temperature: pd.Series) -> pd.DataFrame`

Calculate seasonal temperature statistics.

**Parameters:**

- `temperature`: Daily temperature data

**Returns:**

- DataFrame with statistics by season (DJF, MAM, JJA, SON)

##### `detect_heatwaves(self, temperature: pd.Series, threshold_percentile: float = 90, min_duration: int = 3) -> pd.DataFrame`

Detect heat wave events.

**Parameters:**

- `temperature`: Daily maximum temperature data
- `threshold_percentile`: Percentile for heat wave threshold (default: 90)
- `min_duration`: Minimum duration in days (default: 3)

**Returns:**

- DataFrame with heat wave events and characteristics

**Event Attributes:**

- Start and end dates
- Duration (days)
- Peak temperature
- Mean temperature during event
- Cumulative heat excess

##### `detect_cold_spells(self, temperature: pd.Series, threshold_percentile: float = 10, min_duration: int = 3) -> pd.DataFrame`

Detect cold spell events.

**Parameters:**

- `temperature`: Daily minimum temperature data
- `threshold_percentile`: Percentile for cold threshold (default: 10)
- `min_duration`: Minimum duration in days (default: 3)

**Returns:**

- DataFrame with cold spell events and characteristics

##### `growing_degree_days(self, temperature: pd.Series, base_temp: float = 10.0) -> pd.Series`

Calculate growing degree days.

**Parameters:**

- `temperature`: Daily mean temperature data
- `base_temp`: Base temperature threshold (default: 10°C)

**Returns:**

- Series of daily GDD values

**Formula:**

```
GDD = max(T_mean - T_base, 0)
```

##### `heating_degree_days(self, temperature: pd.Series, base_temp: float = 18.0) -> pd.Series`

Calculate heating degree days.

**Parameters:**

- `temperature`: Daily mean temperature data
- `base_temp`: Base temperature for heating (default: 18°C)

**Returns:**

- Series of daily HDD values

##### `cooling_degree_days(self, temperature: pd.Series, base_temp: float = 22.0) -> pd.Series`

Calculate cooling degree days.

**Parameters:**

- `temperature`: Daily mean temperature data
- `base_temp`: Base temperature for cooling (default: 22°C)

**Returns:**

- Series of daily CDD values

##### `last_spring_frost(self, temperature: pd.Series, frost_threshold: float = 0.0) -> dict`

Find last spring frost date.

**Parameters:**

- `temperature`: Daily minimum temperature data
- `frost_threshold`: Temperature threshold for frost (default: 0°C)

**Returns:**

- Dictionary with frost timing statistics by year

##### `first_autumn_frost(self, temperature: pd.Series, frost_threshold: float = 0.0) -> dict`

Find first autumn frost date.

**Parameters:**

- `temperature`: Daily minimum temperature data
- `frost_threshold`: Temperature threshold for frost (default: 0°C)

**Returns:**

- Dictionary with frost timing statistics by year

##### `frost_free_period(self, temperature: pd.Series, frost_threshold: float = 0.0) -> pd.Series`

Calculate frost-free period length.

**Parameters:**

- `temperature`: Daily minimum temperature data
- `frost_threshold`: Temperature threshold for frost (default: 0°C)

**Returns:**

- Series of annual frost-free period lengths (days)

### meteo/climate_indices.py

#### Class: DroughtIndices

Climate and drought index calculations.

##### `calculate_spi(self, precipitation: pd.Series, scale: int = 3) -> pd.Series`

Calculate Standardized Precipitation Index.

**Parameters:**

- `precipitation`: Daily or monthly precipitation data
- `scale`: Timescale in months (1, 3, 6, 12, 24)

**Returns:**

- Series of SPI values

**SPI Categories:**

- ≥ 2.0: Extremely wet
- 1.5 to 1.99: Very wet  
- 1.0 to 1.49: Moderately wet
- -0.99 to 0.99: Near normal
- -1.0 to -1.49: Moderately dry
- -1.5 to -1.99: Severely dry
- ≤ -2.0: Extremely dry

##### `calculate_spei(self, precipitation: pd.Series, pet: pd.Series, scale: int = 3) -> pd.Series`

Calculate Standardized Precipitation Evapotranspiration Index.

**Parameters:**

- `precipitation`: Daily or monthly precipitation data
- `pet`: Potential evapotranspiration data
- `scale`: Timescale in months

**Returns:**

- Series of SPEI values

**Advantages over SPI:**

- Incorporates temperature effects
- More sensitive to warming trends
- Better for water-limited regions

##### `calculate_pdsi_simplified(self, precipitation: pd.Series, temperature: pd.Series) -> pd.Series`

Calculate simplified Palmer Drought Severity Index.

**Parameters:**

- `precipitation`: Monthly precipitation data
- `temperature`: Monthly temperature data

**Returns:**

- Series of PDSI values

##### `unesco_aridity_index(self, precipitation: pd.Series, pet: pd.Series) -> float`

Calculate UNESCO aridity index.

**Parameters:**

- `precipitation`: Annual precipitation data
- `pet`: Annual potential evapotranspiration data

**Returns:**

- Aridity index value

**Classification:**

- Hyper-arid: < 0.05
- Arid: 0.05 - 0.20
- Semi-arid: 0.20 - 0.50
- Dry sub-humid: 0.50 - 0.65
- Humid: > 0.65

##### `de_martonne_index(self, precipitation: pd.Series, temperature: pd.Series) -> float`

Calculate De Martonne aridity index.

**Parameters:**

- `precipitation`: Annual precipitation sum
- `temperature`: Annual mean temperature

**Returns:**

- De Martonne index value

##### `consecutive_dry_days(self, precipitation: pd.Series, threshold: float = 1.0) -> dict`

Calculate consecutive dry days statistics.

**Parameters:**

- `precipitation`: Daily precipitation data
- `threshold`: Precipitation threshold for "dry" day (default: 1.0 mm)

**Returns:**

- Dictionary with dry day statistics

### meteo/extremes.py

#### Class: ExtremeEvents

Extreme weather event analysis tools.

##### `calculate_thresholds(self, data: pd.Series, percentiles: list = [90, 95, 99]) -> dict`

Calculate percentile-based thresholds.

**Parameters:**

- `data`: Time series data (temperature, precipitation, etc.)
- `percentiles`: List of percentiles to calculate

**Returns:**

- Dictionary with threshold values

##### `detect_extreme_events(self, data: pd.Series, threshold: float, operator: str = 'greater', min_duration: int = 1) -> pd.DataFrame`

Generic extreme event detection.

**Parameters:**

- `data`: Time series data
- `threshold`: Value threshold for event detection
- `operator`: 'greater' or 'less' for threshold comparison
- `min_duration`: Minimum event duration in days

**Returns:**

- DataFrame with detected events

### meteo/seasonal_stats.py

#### Class: SeasonalAnalysis

Seasonal analysis and statistics.

##### `seasonal_means(self, data: pd.Series) -> pd.DataFrame`

Calculate seasonal means.

**Parameters:**

- `data`: Time series data with DatetimeIndex

**Returns:**

- DataFrame with seasonal statistics

##### `monthly_climatology(self, data: pd.Series) -> pd.DataFrame`

Calculate monthly climatology.

**Parameters:**

- `data`: Multi-year time series data

**Returns:**

- DataFrame with monthly statistics (mean, std, min, max, quantiles)

##### `seasonal_anomalies(self, data: pd.Series, reference_period: tuple = None) -> pd.Series`

Calculate seasonal anomalies.

**Parameters:**

- `data`: Time series data
- `reference_period`: Tuple of (start_year, end_year) for baseline

**Returns:**

- Series of seasonal anomaly values

##### `harmonic_analysis(self, data: pd.Series, n_harmonics: int = 3) -> dict`

Perform harmonic analysis of annual cycle.

**Parameters:**

- `data`: Daily time series data
- `n_harmonics`: Number of harmonic components

**Returns:**

- Dictionary with harmonic coefficients and statistics

## Utility Modules

### utils/logger.py

#### Function: `setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger`

Set up a logger with standardized configuration.

**Parameters:**

- `name`: Logger name (typically module name)
- `log_file`: Optional log file path
- `level`: Logging level (default: INFO)

**Returns:**

- Configured logger instance

**Features:**

- Consistent formatting across all modules
- File and console output options
- Error-focused logging strategy
- Module-specific context

**Example:**

```python
from src.utils.logger import setup_logger

logger = setup_logger("my_module", log_file="logs/my_module.log")
logger.info("Processing started")
logger.error("An error occurred: %s", error_message)
```

## Error Handling

### Common Exceptions

#### `ValueError`

Raised for invalid input parameters or data format issues.

#### `IndexError`  

Raised when time series operations fail due to insufficient data.

#### `KeyError`

Raised when required data columns or time periods are missing.

### Best Practices

1. **Input Validation**: Always validate input parameters and data format
2. **Error Logging**: Use the centralized logger for error reporting
3. **Graceful Degradation**: Return NaN or empty results rather than crashing
4. **User Feedback**: Provide clear error messages with suggested solutions

### Example Error Handling

```python
from src.utils.logger import setup_logger

logger = setup_logger("my_analysis")

def safe_analysis(data: pd.Series) -> dict:
    try:
        if data.empty:
            raise ValueError("Input data is empty")
        
        result = perform_analysis(data)
        return result
        
    except ValueError as e:
        logger.error("Analysis failed due to invalid input: %s", e)
        return {}
    except Exception as e:
        logger.error("Unexpected error in analysis: %s", e)
        return {}
```

## Performance Notes

### Optimization Features

1. **Numba JIT Compilation**: Core algorithms compiled for C-like performance
2. **Vectorized Operations**: NumPy/Pandas operations preferred over loops
3. **Memory Efficiency**: Minimal data copying, in-place operations where safe
4. **Parallel Processing**: Multi-core support for independent calculations

### Memory Requirements

- **Typical Usage**: 10-50 MB per year of daily data
- **Large Datasets**: Consider chunked processing for multi-decadal data
- **Peak Memory**: During baseflow separation with reflection

### Processing Speed

- **BFI Calculation**: ~0.1 seconds per year (with JIT compilation)
- **Climate Indices**: ~1-5 seconds per station-year
- **Extreme Events**: ~0.5 seconds per station-year

## Version Compatibility

### Python Requirements

- **Minimum**: Python 3.12
- **Recommended**: Python 3.12+

### Key Dependencies

- **NumPy**: ≥1.24.0
- **Pandas**: ≥2.0.0
- **SciPy**: ≥1.10.0
- **Numba**: ≥0.57.0
- **Matplotlib**: ≥3.7.0

### Breaking Changes

Future versions may include breaking changes:

- Function signature modifications
- Return value format changes
- Default parameter updates

All breaking changes will be documented in release notes.
