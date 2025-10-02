# Meteorological Analysis and Climate Indices

This document describes the meteorological analysis tools and climate indices implemented in the CAMELS-RU project.

## Temperature Analysis

### Basic Statistics

The `TemperatureAnalysis` class provides comprehensive temperature statistics:

```python
from src.meteo.temperature import TemperatureAnalysis

temp_analyzer = TemperatureAnalysis()
stats = temp_analyzer.basic_statistics(temperature_data)
```

#### Statistical Measures

**Central Tendency:**

- Mean temperature
- Median temperature  
- Modal temperature (most frequent)

**Variability:**

- Standard deviation
- Variance
- Interquartile range (IQR)
- Coefficient of variation

**Distribution Shape:**

- Skewness (asymmetry measure)
- Kurtosis (tail heaviness)
- Percentiles (1st, 5th, 10th, 25th, 75th, 90th, 95th, 99th)

### Extreme Temperature Events

#### Heat Wave Detection

Heat waves are identified using multiple criteria:

```python
def detect_heatwaves(self, temperature: pd.Series, 
                    threshold_percentile: float = 90,
                    min_duration: int = 3) -> pd.DataFrame:
```

**Methodology:**

1. Calculate threshold as nth percentile of historical data
2. Identify consecutive days above threshold
3. Filter events by minimum duration
4. Calculate intensity metrics for each event

**Event Characteristics:**

- Start and end dates
- Duration (days)
- Maximum temperature
- Mean temperature during event
- Cumulative heat excess

#### Cold Spell Detection

Similar methodology applied for cold extremes:

```python
def detect_cold_spells(self, temperature: pd.Series,
                      threshold_percentile: float = 10,
                      min_duration: int = 3) -> pd.DataFrame:
```

### Degree Day Calculations

#### Growing Degree Days (GDD)

Used for agricultural and ecological applications:

```python
def growing_degree_days(self, temperature: pd.Series,
                       base_temp: float = 10.0) -> pd.Series:
    return np.maximum(temperature - base_temp, 0)
```

**Applications:**

- Crop development modeling
- Phenological studies
- Vegetation growth prediction

#### Heating Degree Days (HDD)

Energy demand estimation:

```python
def heating_degree_days(self, temperature: pd.Series,
                       base_temp: float = 18.0) -> pd.Series:
    return np.maximum(base_temp - temperature, 0)
```

#### Cooling Degree Days (CDD)

Air conditioning requirements:

```python
def cooling_degree_days(self, temperature: pd.Series,
                       base_temp: float = 22.0) -> pd.Series:
    return np.maximum(temperature - base_temp, 0)
```

### Frost Analysis

#### Last Spring Frost

```python
def last_spring_frost(self, temperature: pd.Series,
                     frost_threshold: float = 0.0) -> dict:
```

Identifies the last occurrence of temperature below threshold in spring (March-May).

#### First Autumn Frost

```python
def first_autumn_frost(self, temperature: pd.Series,
                      frost_threshold: float = 0.0) -> dict:
```

Identifies the first occurrence of temperature below threshold in autumn (September-November).

#### Frost-Free Period

Calculates the growing season length between last spring and first autumn frost.

## Climate Indices and Drought Indicators

### Standardized Precipitation Index (SPI)

The SPI quantifies precipitation deficits on multiple timescales:

```python
from src.meteo.climate_indices import DroughtIndices

drought_analyzer = DroughtIndices()
spi_values = drought_analyzer.calculate_spi(precipitation_data, scale=3)
```

#### Mathematical Formulation

1. **Aggregation**: Sum precipitation over specified timescale (1, 3, 6, 12, 24 months)
2. **Fitting**: Fit gamma distribution to aggregated data
3. **Transformation**: Convert to standard normal distribution

**Gamma Distribution Parameters:**

```python
# Shape parameter (α)
alpha = (mean_precip / std_precip) ** 2

# Scale parameter (β)  
beta = std_precip ** 2 / mean_precip

# Probability density function
pdf = (x**(alpha-1) * np.exp(-x/beta)) / (beta**alpha * gamma(alpha))
```

#### SPI Classification

| SPI Value | Category | Probability (%) |
|-----------|----------|-----------------|
| ≥ 2.0 | Extremely wet | 2.3 |
| 1.5 to 1.99 | Very wet | 4.4 |
| 1.0 to 1.49 | Moderately wet | 9.2 |
| -0.99 to 0.99 | Near normal | 68.2 |
| -1.0 to -1.49 | Moderately dry | 9.2 |
| -1.5 to -1.99 | Severely dry | 4.4 |
| ≤ -2.0 | Extremely dry | 2.3 |

### Standardized Precipitation Evapotranspiration Index (SPEI)

SPEI incorporates both precipitation and potential evapotranspiration:

```python
def calculate_spei(self, precipitation: pd.Series,
                  pet: pd.Series, scale: int = 3) -> pd.Series:
    # Water balance
    wb = precipitation - pet
    
    # Aggregate over timescale
    wb_agg = wb.rolling(window=scale).sum()
    
    # Fit log-logistic distribution
    return self._fit_log_logistic(wb_agg)
```

#### Advantages over SPI

- Accounts for temperature effects via evapotranspiration
- More sensitive to warming trends
- Better representation of drought in water-limited regions

### Palmer Drought Severity Index (PDSI)

Simplified implementation of Palmer's methodology:

```python
def calculate_pdsi_simplified(self, precipitation: pd.Series,
                            temperature: pd.Series) -> pd.Series:
```

**Components:**

1. **Potential Evapotranspiration**: Thornthwaite method
2. **Water Balance**: Monthly accounting of moisture supply and demand
3. **Drought Severity**: Standardized departure from normal conditions

### Aridity Indices

#### UNESCO Aridity Index

```python
def unesco_aridity_index(self, precipitation: pd.Series,
                        pet: pd.Series) -> float:
    return precipitation.sum() / pet.sum()
```

**Classification:**

- Hyper-arid: < 0.05
- Arid: 0.05 - 0.20
- Semi-arid: 0.20 - 0.50
- Dry sub-humid: 0.50 - 0.65
- Humid: > 0.65

#### De Martonne Aridity Index

```python
def de_martonne_index(self, precipitation: pd.Series,
                     temperature: pd.Series) -> float:
    annual_precip = precipitation.sum()
    mean_temp = temperature.mean()
    return annual_precip / (mean_temp + 10)
```

### Consecutive Dry Days

Identifies longest periods without significant precipitation:

```python
def consecutive_dry_days(self, precipitation: pd.Series,
                        threshold: float = 1.0) -> dict:
```

**Methodology:**

1. Define dry day as precipitation < threshold
2. Identify consecutive dry periods
3. Calculate statistics for each year and overall period

## Extreme Event Analysis

### Percentile-Based Thresholds

```python
from src.meteo.extremes import ExtremeEvents

extreme_analyzer = ExtremeEvents()
thresholds = extreme_analyzer.calculate_thresholds(data, percentiles=[90, 95, 99])
```

### Heat Wave Characteristics

**Intensity Measures:**

- Peak temperature during event
- Mean temperature during event
- Cumulative heat excess above threshold
- Heat wave magnitude index

**Duration Metrics:**

- Event length in days
- Consecutive hours above threshold
- Time to peak intensity

### Precipitation Extremes

**Daily Extremes:**

- Maximum 1-day precipitation
- Maximum 5-day precipitation
- Number of days above percentile thresholds

**Intensity-Duration-Frequency (IDF) Analysis:**

```python
def idf_analysis(self, precipitation: pd.Series,
                durations: list = [1, 3, 5, 10],
                return_periods: list = [2, 5, 10, 25, 50, 100]) -> pd.DataFrame:
```

## Seasonal Analysis

### Seasonal Statistics

```python
from src.meteo.seasonal_stats import SeasonalAnalysis

seasonal_analyzer = SeasonalAnalysis()
seasonal_means = seasonal_analyzer.seasonal_means(data)
```

#### Standard Seasons

- **Winter**: December, January, February (DJF)
- **Spring**: March, April, May (MAM)  
- **Summer**: June, July, August (JJA)
- **Autumn**: September, October, November (SON)

### Monthly Climatology

Calculates long-term monthly statistics:

```python
def monthly_climatology(self, data: pd.Series) -> pd.DataFrame:
    return data.groupby(data.index.month).agg([
        'mean', 'std', 'min', 'max', 
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75)
    ])
```

### Annual Cycle Analysis

#### Harmonic Analysis

Fits sinusoidal functions to capture annual cycles:

```python
def harmonic_analysis(self, data: pd.Series, n_harmonics: int = 3) -> dict:
    # Fit: y = a0 + Σ(ai*cos(i*ωt) + bi*sin(i*ωt))
    # where ω = 2π/365.25
```

**Applications:**

- Peak timing identification
- Amplitude quantification
- Phase shift detection
- Climate change trend analysis

### Seasonal Anomalies

```python
def seasonal_anomalies(self, data: pd.Series,
                      reference_period: tuple = None) -> pd.Series:
    # Calculate departures from long-term seasonal means
```

## Quality Control and Validation

### Data Requirements

**Temporal Coverage:**

- Minimum: 30 years for robust climatology
- Recommended: 50+ years for trend analysis
- Missing data: < 10% for reliable statistics

**Quality Flags:**

- Range checks (physical limits)
- Temporal consistency tests
- Spatial coherence validation
- Outlier detection and flagging

### Uncertainty Quantification

**Bootstrap Confidence Intervals:**

```python
def bootstrap_confidence_interval(self, data: pd.Series,
                                 statistic: callable,
                                 n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> tuple:
```

**Sensitivity Analysis:**

- Parameter perturbation studies
- Alternative threshold testing
- Cross-validation with independent datasets

## Implementation Notes

### Performance Optimization

- Vectorized operations using NumPy/Pandas
- Efficient rolling window calculations
- Memory-conscious processing for large datasets
- Parallel processing for multiple stations

### Validation Status

All climate indices validated against:

- WMO standards and guidelines
- Published research implementations
- International climate monitoring centers
- Cross-comparison with established software packages

## References

1. McKee, T. B., Doesken, N. J., & Kleist, J. (1993). The relationship of drought frequency and duration to time scales. *Proceedings of the 8th Conference on Applied Climatology*, 17(22), 179-183.

2. Vicente-Serrano, S. M., Beguería, S., & López-Moreno, J. I. (2010). A multiscalar drought index sensitive to global warming: the standardized precipitation evapotranspiration index. *Journal of Climate*, 23(7), 1696-1718.

3. Palmer, W. C. (1965). Meteorological drought. *US Department of Commerce, Weather Bureau Research Paper*, 45.

4. Thornthwaite, C. W. (1948). An approach toward a rational classification of climate. *Geographical Review*, 38(1), 55-94.

5. Alexander, L. V., et al. (2006). Global observed changes in daily climate extremes of temperature and precipitation. *Journal of Geophysical Research*, 111(D5).