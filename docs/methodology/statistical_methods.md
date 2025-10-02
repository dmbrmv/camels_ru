# Time Series Analysis and Statistical Methods

This document describes the statistical analysis methods and time series techniques implemented in CAMELS-RU.

## Time Series Statistics Package

### Trend Analysis

The `TrendAnalysis` class provides comprehensive trend detection and characterization methods.

#### Mann-Kendall Trend Test

**Non-parametric test for monotonic trends:**

```python
from src.timeseries_stats.trends import TrendAnalysis

trend_analyzer = TrendAnalysis()
result = trend_analyzer.mann_kendall_test(time_series)
```

**Mathematical Formulation:**

```
S = Σ(i=1 to n-1) Σ(j=i+1 to n) sign(x_j - x_i)

where sign(θ) = {
    +1 if θ > 0
     0 if θ = 0
    -1 if θ < 0
}
```

**Test Statistic:**

```
Z = {
    (S-1)/√Var(S)     if S > 0
    0                 if S = 0
    (S+1)/√Var(S)     if S < 0
}
```

**Variance Calculation:**

```
Var(S) = n(n-1)(2n+5)/18
```

**Output:**

- Kendall's tau (correlation coefficient)
- Z-statistic and p-value
- Trend direction and significance
- Slope estimate (Sen's method)

#### Sen's Slope Estimator

**Robust slope estimation:**

```python
slope = trend_analyzer.sens_slope(time_series)
```

**Formula:**

```
slope = median{(x_j - x_i)/(j - i)} for all i < j
```

**Advantages:**

- Insensitive to outliers
- No distributional assumptions
- Suitable for non-linear trends

#### Seasonal Mann-Kendall Test

**Accounts for seasonal patterns:**

```python
seasonal_result = trend_analyzer.seasonal_mann_kendall(
    time_series, 
    season_length=12
)
```

**Application:**

- Monthly data with annual cycles
- Weekly data with annual patterns
- Sub-annual seasonal variations

#### Theil-Sen Regression

**Robust linear regression:**

```python
slope, intercept, r_value = trend_analyzer.theil_sen_regression(x, y)
```

**Method:**

1. Calculate slopes for all point pairs
2. Use median slope as estimate
3. Calculate intercept through data median

#### Piecewise Linear Regression

**Change point detection:**

```python
change_points = trend_analyzer.piecewise_regression(
    time_series,
    max_segments=3,
    min_segment_length=10
)
```

**Applications:**

- Climate shift detection
- Regime change identification
- Multi-phase trend analysis

### Statistical Measures

#### Central Tendency

**Mean (Arithmetic):**

```python
mean = np.mean(data)
```

**Median (Robust):**

```python
median = np.median(data)
```

**Trimmed Mean (Outlier-resistant):**

```python
trimmed_mean = stats.trim_mean(data, proportiontocut=0.1)
```

#### Variability Measures

**Standard Deviation:**

```python
std = np.std(data, ddof=1)  # Sample standard deviation
```

**Coefficient of Variation:**

```python
cv = std / mean
```

**Interquartile Range:**

```python
q75, q25 = np.percentile(data, [75, 25])
iqr = q75 - q25
```

**Median Absolute Deviation (Robust):**

```python
mad = np.median(np.abs(data - np.median(data)))
```

#### Distribution Shape

**Skewness:**

```python
skewness = stats.skew(data)
```

**Interpretation:**

- Skewness > 0: Right-tailed distribution
- Skewness < 0: Left-tailed distribution
- |Skewness| < 0.5: Approximately symmetric

**Kurtosis:**

```python
kurtosis = stats.kurtosis(data)
```

**Interpretation:**

- Kurtosis > 0: Heavy-tailed (leptokurtic)
- Kurtosis < 0: Light-tailed (platykurtic)
- Kurtosis ≈ 0: Normal-like tails (mesokurtic)

### Autocorrelation Analysis

#### Autocorrelation Function (ACF)

**Measures temporal dependence:**

```python
def autocorrelation_function(data, max_lag=50):
    autocorr = np.correlate(data, data, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]  # Normalize
    return autocorr[:max_lag+1]
```

**Applications:**

- Identify cyclical patterns
- Assess model residuals
- Determine memory length

#### Partial Autocorrelation Function (PACF)

**Direct correlation at specific lags:**

```python
from statsmodels.tsa.stattools import pacf

pacf_values = pacf(data, nlags=max_lag)
```

**Uses:**

- AR model order selection
- Remove indirect correlations
- Identify significant lags

#### Ljung-Box Test

**Test for autocorrelation:**

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

ljung_box_result = acorr_ljungbox(residuals, lags=10)
```

**Null Hypothesis:** No autocorrelation in residuals

### Spectral Analysis

#### Periodogram

**Frequency domain analysis:**

```python
from scipy.signal import periodogram

frequencies, power = periodogram(data, fs=1.0)
```

**Applications:**

- Identify dominant cycles
- Detect periodic signals
- Compare frequency content

#### Welch's Method

**Improved spectral estimation:**

```python
from scipy.signal import welch

frequencies, power = welch(data, fs=1.0, nperseg=256)
```

**Advantages:**

- Reduced variance
- Better frequency resolution
- Overlapping windows

### Change Point Detection

#### CUSUM Test

**Cumulative sum control chart:**

```python
def cusum_test(data, threshold=5):
    mean_data = np.mean(data)
    cumsum_pos = np.maximum.accumulate(
        np.maximum(data - mean_data - threshold, 0)
    )
    cumsum_neg = np.maximum.accumulate(
        np.maximum(-data + mean_data - threshold, 0)
    )
    return cumsum_pos, cumsum_neg
```

#### PELT Algorithm

**Pruned Exact Linear Time:**

```python
# Future implementation
def pelt_changepoint_detection(data, penalty='BIC'):
    # Efficient change point detection
    pass
```

### Missing Data Analysis

#### Missing Data Patterns

**Identify missingness structure:**

```python
def analyze_missing_patterns(data):
    missing_count = data.isnull().sum()
    missing_percent = (missing_count / len(data)) * 100
    
    return {
        'total_missing': missing_count.sum(),
        'percent_missing': missing_percent,
        'longest_gap': find_longest_gap(data),
        'gap_distribution': gap_length_distribution(data)
    }
```

#### Gap Filling Methods

**Linear Interpolation:**

```python
filled_data = data.interpolate(method='linear')
```

**Spline Interpolation:**

```python
filled_data = data.interpolate(method='spline', order=3)
```

**Forward/Backward Fill:**

```python
filled_data = data.fillna(method='ffill').fillna(method='bfill')
```

**Seasonal Decomposition Interpolation:**

```python
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(data, model='additive')
# Fill gaps in each component separately
```

### Quality Control Statistics

#### Outlier Detection

**Z-Score Method:**

```python
def zscore_outliers(data, threshold=3):
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    return data[z_scores > threshold]
```

**Modified Z-Score (Robust):**

```python
def modified_zscore_outliers(data, threshold=3.5):
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return data[np.abs(modified_z_scores) > threshold]
```

**Interquartile Range Method:**

```python
def iqr_outliers(data, factor=1.5):
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    lower_bound = q25 - factor * iqr
    upper_bound = q75 + factor * iqr
    return data[(data < lower_bound) | (data > upper_bound)]
```

#### Data Quality Metrics

**Completeness:**

```python
completeness = (1 - data.isnull().sum() / len(data)) * 100
```

**Consistency:**

```python
def temporal_consistency(data, max_change_rate=0.1):
    daily_changes = np.abs(data.diff() / data.shift(1))
    inconsistent = daily_changes > max_change_rate
    return (1 - inconsistent.sum() / len(data)) * 100
```

**Plausibility:**

```python
def physical_plausibility(data, valid_range):
    within_range = (data >= valid_range[0]) & (data <= valid_range[1])
    return (within_range.sum() / len(data)) * 100
```

### Extreme Value Analysis

#### Generalized Extreme Value (GEV) Distribution

**Annual maxima analysis:**

```python
from scipy.stats import genextreme

# Fit GEV distribution
shape, loc, scale = genextreme.fit(annual_maxima)

# Calculate return levels
return_periods = [2, 5, 10, 25, 50, 100]
return_levels = genextreme.ppf(
    1 - 1/np.array(return_periods), 
    shape, loc, scale
)
```

#### Peaks Over Threshold (POT)

**Generalized Pareto Distribution:**

```python
from scipy.stats import genpareto

# Define threshold (e.g., 95th percentile)
threshold = np.percentile(data, 95)
exceedances = data[data > threshold] - threshold

# Fit GPD
shape, loc, scale = genpareto.fit(exceedances)
```

#### Block Maxima Method

**Extract extreme values:**

```python
def block_maxima(data, block_size='A'):
    """Extract maximum values from blocks."""
    if isinstance(data.index, pd.DatetimeIndex):
        return data.resample(block_size).max()
    else:
        # For regular integer index
        blocks = np.array_split(data, len(data) // block_size)
        return np.array([block.max() for block in blocks])
```

### Model Diagnostics

#### Residual Analysis

**Normality Tests:**

```python
from scipy.stats import shapiro, normaltest, anderson

# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(residuals)

# D'Agostino-Pearson test
dagostino_stat, dagostino_p = normaltest(residuals)

# Anderson-Darling test
anderson_result = anderson(residuals, dist='norm')
```

**Homoscedasticity Tests:**

```python
from statsmodels.stats.diagnostic import het_breuschpagan

# Breusch-Pagan test
bp_stat, bp_p, f_stat, f_p = het_breuschpagan(residuals, exog)
```

#### Model Selection Criteria

**Information Criteria:**

```python
def calculate_aic(residuals, n_params):
    n = len(residuals)
    sse = np.sum(residuals**2)
    aic = 2*n_params + n*np.log(sse/n)
    return aic

def calculate_bic(residuals, n_params):
    n = len(residuals)
    sse = np.sum(residuals**2)
    bic = n_params*np.log(n) + n*np.log(sse/n)
    return bic
```

### Cross-Validation Techniques

#### Time Series Cross-Validation

**Forward Chaining:**

```python
def time_series_cv(data, min_train_size, horizon):
    splits = []
    for i in range(min_train_size, len(data) - horizon + 1):
        train_idx = slice(0, i)
        test_idx = slice(i, i + horizon)
        splits.append((train_idx, test_idx))
    return splits
```

#### Walk-Forward Validation

**Expanding window approach:**

```python
def walk_forward_validation(data, model_func, window_size):
    predictions = []
    for i in range(window_size, len(data)):
        train_data = data[i-window_size:i]
        model = model_func(train_data)
        pred = model.predict(1)  # One-step ahead
        predictions.append(pred)
    return np.array(predictions)
```

## Performance Metrics

### Forecast Accuracy

**Mean Absolute Error (MAE):**

```python
mae = np.mean(np.abs(predictions - observations))
```

**Root Mean Squared Error (RMSE):**

```python
rmse = np.sqrt(np.mean((predictions - observations)**2))
```

**Nash-Sutcliffe Efficiency (NSE):**

```python
nse = 1 - np.sum((predictions - observations)**2) / \
          np.sum((observations - np.mean(observations))**2)
```

**Kling-Gupta Efficiency (KGE):**

```python
def kling_gupta_efficiency(predictions, observations):
    r = np.corrcoef(predictions, observations)[0, 1]
    alpha = np.std(predictions) / np.std(observations)
    beta = np.mean(predictions) / np.mean(observations)
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    return kge
```

## Implementation Notes

### Computational Efficiency

**Vectorization:**

- Use NumPy operations over Python loops
- Leverage pandas built-in functions
- Avoid explicit iteration when possible

**Memory Management:**

- Process data in chunks for large datasets
- Use generators for sequential processing
- Clear intermediate variables

**Parallel Processing:**

- Use joblib for embarrassingly parallel tasks
- Apply multiprocessing for independent calculations
- Consider Dask for distributed computing

### Numerical Stability

**Precision Considerations:**

- Use double precision (float64) for critical calculations
- Check for numerical overflow/underflow
- Implement robust algorithms for ill-conditioned problems

**Convergence Criteria:**

- Set appropriate tolerance levels
- Implement maximum iteration limits
- Provide convergence diagnostics

## References

1. Kendall, M. G. (1975). Rank Correlation Methods. Charles Griffin, London.

2. Sen, P. K. (1968). Estimates of the regression coefficient based on Kendall's tau. Journal of the American Statistical Association, 63(324), 1379-1389.

3. Theil, H. (1950). A rank-invariant method of linear and polynomial regression analysis. Nederlandse Akademie van Wetenschappen, 53, 386-392.

4. Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297-303.

5. Page, E. S. (1954). Continuous inspection schemes. Biometrika, 41(1/2), 100-115.
