# Hydrological Metrics and Signatures

This document provides comprehensive documentation for all hydrological metrics and signatures implemented in the CAMELS-RU project.

## Base Flow Index (BFI)

### Mathematical Formulation

The Base Flow Index (BFI) is calculated using the Eckhardt (2005) digital filter method with Monte Carlo uncertainty estimation.

#### Algorithm Description

The Eckhardt filter performs a three-pass filtering process:

1. **First Pass (Forward)**: Initial quick flow separation
2. **Backward Pass**: Refinement using backward iteration
3. **Forward Pass**: Final adjustment for smooth baseflow

#### Mathematical Equations

**First Pass:**

```
q_f[i+1] = α × q_f[i] + 0.5 × (1 + α) × (Q[i+1] - Q[i])
q_b[i] = Q[i] - q_f[i] if q_f[i] > 0, else Q[i]
```

**Backward Pass:**

```
q_f[i] = α × q_f[i+1] + 0.5 × (1 + α) × (Q[i] - Q[i+1])
q_b[i] = Q[i] - q_f[i] if q_f[i] > 0, else Q[i]
```

Where:

- `Q[i]` = observed discharge at time i
- `q_f[i]` = quick flow component
- `q_b[i]` = base flow component  
- `α` = filter parameter (0.9-0.98)

#### Monte Carlo Implementation

The BFI calculation uses 1000 random α values between 0.9 and 0.98 to account for parameter uncertainty:

```python
@numba.jit(nopython=True)
def bfi_1000(Q, passes, reflect):
    random.seed(1996)  # Reproducible results
    alpha_coefficients = [random.uniform(0.9, 0.98) for i in range(1000)]
    # Process for each α and return mean BFI
```

#### Edge Effect Mitigation

To reduce initialization bias, the algorithm:

1. Reflects the first `reflect` values at the beginning
2. Reflects the last `reflect` values at the end
3. Crops the reflected portions after filtering

**Default Parameters:**

- Number of passes: 3
- Reflection length: 30 days
- Random seed: 1996 (for reproducibility)

### Implementation Details

#### Function Signatures

```python
def bfi(Q: np.ndarray, alpha: float, passes: int, reflect: int) -> tuple[float, np.ndarray]
def bfi_1000(Q: np.ndarray, passes: int, reflect: int) -> tuple[float, np.ndarray]
```

#### Performance Optimization

- All core functions are JIT-compiled with Numba for C-like performance
- Memory-efficient implementation using numpy arrays
- Vectorized operations where possible

## Flow Duration Curve Metrics

### Slope of Flow Duration Curve

Measures the variability of flow using percentiles:

```python
def slope_fdc_gauge(hydro_year: pd.Series) -> float:
    slope_fdc = (
        math.log(np.nanpercentile(hydro_year, q=67)) -
        math.log(np.nanpercentile(hydro_year, q=33))
    ) / (0.67 - 0.33)
    return slope_fdc
```

**Physical Meaning:**

- Higher values indicate more variable flow regimes
- Lower values suggest more stable, groundwater-dominated systems

## Flow Timing Metrics

### Half Flow Date (HFD)

Date when cumulative discharge reaches 50% of annual total:

```python
def hfd_calc(calendar_year: pd.Series, hydro_year: pd.Series) -> pd.Timestamp:
    cal_val = np.nansum(calendar_year) / 2
    return hydro_year[hydro_year.cumsum() > cal_val].index[0]
```

**Applications:**

- Snowmelt timing assessment
- Climate change impact detection
- Regional flow regime classification

## Flow Extremes

### Quantile-Based Metrics

```python
def q5_q95(hydro_year: pd.Series) -> dict[str, float]:
    q5 = np.nanpercentile(hydro_year, q=95)   # High flow (5% exceedance)
    q95 = np.nanpercentile(hydro_year, q=5)   # Low flow (95% exceedance)
    return {'q5': q5, 'q95': q95}
```

### High Flow Analysis

**High Flow Frequency:**

```python
def high_q_freq(hydro_year: pd.Series) -> pd.Series:
    # Days with flow > 9 × median
    med_val = np.nanmedian(hydro_year) * 9
    return hydro_year[hydro_year > med_val]
```

**High Flow Duration:**

```python
def high_q_dur(hydro_year: pd.Series) -> list[pd.Series]:
    # Continuous periods above 2 × median
    mean_lim = np.nanmedian(hydro_year) * 2
    return [periods above threshold]
```

### Low Flow Analysis

**Low Flow Frequency:**

```python
def low_q_freq(hydro_year: pd.Series) -> pd.Series:
    # Days with flow < 0.2 × mean
    mean_val = np.nanmean(hydro_year) * 0.2
    return hydro_year[hydro_year < mean_val]
```

**Low Flow Duration:**

```python
def low_q_dur(hydro_year: pd.Series) -> list[pd.Series]:
    # Continuous periods below mean
    return [periods below mean flow]
```

## Data Processing Requirements

### Input Data Format

- **Time Series**: Pandas Series with DatetimeIndex
- **Units**: Discharge in m³/s (cubic meters per second)
- **Frequency**: Daily observations
- **Quality**: No missing values within analysis period

### Gap Handling

The current implementation requires continuous data:

- Series with gaps are split into sub-series
- Each continuous segment is analyzed separately
- Minimum recommended length: 300 days

### Hydrological Year Definition

Following standard hydrological practice:

- **Start**: October 1st
- **End**: September 30th (following year)
- **Rationale**: Captures complete annual cycle in most climates

## Quality Control

### Data Validation

1. **Range Checks**: Discharge values must be non-negative
2. **Temporal Consistency**: Monotonic time index required
3. **Minimum Length**: At least 300 observations for reliable statistics
4. **Physical Plausibility**: Extreme value detection and flagging

### Uncertainty Quantification

- **BFI Uncertainty**: Reported through Monte Carlo range
- **Temporal Stability**: Multi-year analysis for trend detection
- **Sensitivity Analysis**: Parameter perturbation studies

## References

1. Eckhardt, K. (2005). How to construct recursive digital filters for baseflow separation. *Hydrological Processes*, 19(2), 507-515.

2. Addor, N., Nearing, G., Prieto, C., Newman, A. J., Le Vine, N., & Clark, M. P. (2018). A ranking of hydrological signatures based on their predictability in space. *Water Resources Research*, 54(11), 8792-8812.

3. Nathan, R. J., & McMahon, T. A. (1990). Evaluation of automated techniques for base flow and recession analyses. *Water Resources Research*, 26(7), 1465-1473.

4. Ladson, A. R., Brown, R., Neal, B., & Nathan, R. (2013). A standard approach to baseflow separation using the Lyne and Hollick filter. *Australian Journal of Water Resources*, 17(1), 25-34.

## Implementation Notes

### Performance Considerations

- Numba JIT compilation provides 10-100x speedup over pure Python
- Memory usage scales linearly with time series length
- Typical processing time: <1 second per year of daily data

### Validation Status

All metrics have been validated against:

- Published literature examples
- Standard hydrological software (e.g., EflowStats, hydrostats)
- Manual calculations for selected test cases

### Future Enhancements

Planned improvements include:

- Additional baseflow separation methods (Lyne-Hollick, UKIH)
- Recession curve analysis
- Flow regime classification algorithms
- Seasonal signature variants
