# Practical Usage Guide

This guide provides practical examples and workflows for using the CAMELS-RU analysis tools.

## Quick Start Workflow

### 1. Basic Hydrological Analysis

```python
import pandas as pd
import numpy as np
from src.hydro_metrics import hydro_job, bfi_1000
from src.processing import split_by_hydro_year

# Load discharge data
discharge = pd.read_csv("discharge.csv", index_col=0, parse_dates=True)

# Split by hydrological years
hydro_years = split_by_hydro_year(discharge['flow'])

# Analyze each year
results = {}
for year, data in hydro_years.items():
    if len(data) > 300:  # Minimum data requirement
        # Calculate base flow index
        bfi_value, baseflow = bfi_1000(data.values, passes=3, reflect=30)
        
        # Store results
        results[year] = {
            'bfi': bfi_value,
            'mean_flow': data.mean(),
            'flow_range': data.max() - data.min(),
            'data_completeness': len(data) / 365.25
        }
        
        # Generate visualization
        hydro_job(data, data)

# Summary statistics
bfi_values = [r['bfi'] for r in results.values()]
print(f"Mean BFI: {np.mean(bfi_values):.3f}")
print(f"BFI range: {np.min(bfi_values):.3f} - {np.max(bfi_values):.3f}")
```

### 2. Climate Index Analysis

```python
from src.meteo.climate_indices import DroughtIndices
from src.meteo.temperature import TemperatureAnalysis

# Load meteorological data
precip = pd.read_csv("precipitation.csv", index_col=0, parse_dates=True)
temp = pd.read_csv("temperature.csv", index_col=0, parse_dates=True)

# Initialize analyzers
drought_analyzer = DroughtIndices()
temp_analyzer = TemperatureAnalysis()

# Calculate drought indices
spi_3 = drought_analyzer.calculate_spi(precip['precip'], scale=3)
spi_12 = drought_analyzer.calculate_spi(precip['precip'], scale=12)

# Temperature analysis
temp_stats = temp_analyzer.basic_statistics(temp['temp_mean'])
heatwaves = temp_analyzer.detect_heatwaves(temp['temp_max'])

print(f"Temperature statistics:")
print(f"  Mean: {temp_stats['mean']:.1f}°C")
print(f"  Range: {temp_stats['range']:.1f}°C")
print(f"  Heat waves detected: {len(heatwaves)}")
```

## Advanced Analysis Workflows

### Multi-Station Comparison

```python
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_multiple_stations(data_directory):
    """Analyze multiple stations and compare results."""
    results = {}
    
    # Process each station
    for station_file in Path(data_directory).glob("station_*.csv"):
        station_id = station_file.stem.split('_')[1]
        
        # Load and process data
        discharge = pd.read_csv(station_file, index_col=0, parse_dates=True)
        hydro_years = split_by_hydro_year(discharge['flow'])
        
        # Calculate metrics for each year
        station_results = []
        for year, data in hydro_years.items():
            if len(data) > 300:
                bfi_val, _ = bfi_1000(data.values, 3, 30)
                station_results.append({
                    'year': int(year),
                    'bfi': bfi_val,
                    'mean_flow': data.mean(),
                    'cv': data.std() / data.mean()
                })
        
        results[station_id] = pd.DataFrame(station_results)
    
    return results

# Example usage
station_results = analyze_multiple_stations("data/discharge/")

# Compare BFI across stations
fig, ax = plt.subplots(figsize=(12, 6))
for station_id, data in station_results.items():
    ax.plot(data['year'], data['bfi'], label=f'Station {station_id}')

ax.set_xlabel('Year')
ax.set_ylabel('Base Flow Index')
ax.legend()
ax.grid(True, alpha=0.3)
plt.title('BFI Comparison Across Stations')
plt.show()
```

### Seasonal Analysis

```python
from src.meteo.seasonal_stats import SeasonalAnalysis

def comprehensive_seasonal_analysis(data):
    """Perform comprehensive seasonal analysis."""
    seasonal_analyzer = SeasonalAnalysis()
    
    # Calculate seasonal statistics
    seasonal_means = seasonal_analyzer.seasonal_means(data)
    monthly_clim = seasonal_analyzer.monthly_climatology(data)
    anomalies = seasonal_analyzer.seasonal_anomalies(data)
    
    # Harmonic analysis for annual cycle
    harmonic_results = seasonal_analyzer.harmonic_analysis(data)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Seasonal means
    seasonal_means.plot(kind='bar', ax=axes[0,0])
    axes[0,0].set_title('Seasonal Means')
    axes[0,0].set_ylabel('Value')
    
    # Monthly climatology
    monthly_clim['mean'].plot(ax=axes[0,1])
    axes[0,1].fill_between(
        monthly_clim.index,
        monthly_clim['mean'] - monthly_clim['std'],
        monthly_clim['mean'] + monthly_clim['std'],
        alpha=0.3
    )
    axes[0,1].set_title('Monthly Climatology')
    axes[0,1].set_ylabel('Value')
    
    # Seasonal anomalies
    anomalies.plot(ax=axes[1,0])
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1,0].set_title('Seasonal Anomalies')
    axes[1,0].set_ylabel('Anomaly')
    
    # Annual cycle harmonics
    t = np.arange(1, 366)
    fitted_cycle = harmonic_results['mean'] + sum(
        harmonic_results[f'harmonic_{i}'] 
        for i in range(1, 4)
    )
    axes[1,1].plot(t, fitted_cycle)
    axes[1,1].set_title('Fitted Annual Cycle')
    axes[1,1].set_xlabel('Day of Year')
    axes[1,1].set_ylabel('Value')
    
    plt.tight_layout()
    return seasonal_means, monthly_clim, anomalies, harmonic_results
```

### Extreme Event Analysis

```python
from src.meteo.extremes import ExtremeEvents

def extreme_event_analysis(temperature_data, precipitation_data):
    """Comprehensive extreme event analysis."""
    extreme_analyzer = ExtremeEvents()
    
    # Temperature extremes
    temp_thresholds = extreme_analyzer.calculate_thresholds(
        temperature_data, 
        percentiles=[10, 90, 95, 99]
    )
    
    heat_waves = extreme_analyzer.detect_extreme_events(
        temperature_data,
        threshold=temp_thresholds[90],
        operator='greater',
        min_duration=3
    )
    
    cold_spells = extreme_analyzer.detect_extreme_events(
        temperature_data,
        threshold=temp_thresholds[10],
        operator='less',
        min_duration=3
    )
    
    # Precipitation extremes
    precip_thresholds = extreme_analyzer.calculate_thresholds(
        precipitation_data,
        percentiles=[95, 99, 99.9]
    )
    
    heavy_precip = extreme_analyzer.detect_extreme_events(
        precipitation_data,
        threshold=precip_thresholds[95],
        operator='greater',
        min_duration=1
    )
    
    # Summary statistics
    print(f"Extreme Event Summary:")
    print(f"  Heat waves: {len(heat_waves)} events")
    print(f"  Cold spells: {len(cold_spells)} events")
    print(f"  Heavy precipitation: {len(heavy_precip)} events")
    
    if len(heat_waves) > 0:
        print(f"  Longest heat wave: {heat_waves['duration'].max()} days")
        print(f"  Mean heat wave duration: {heat_waves['duration'].mean():.1f} days")
    
    return {
        'heat_waves': heat_waves,
        'cold_spells': cold_spells,
        'heavy_precipitation': heavy_precip,
        'thresholds': {
            'temperature': temp_thresholds,
            'precipitation': precip_thresholds
        }
    }
```

## Quality Control Workflows

### Data Validation Pipeline

```python
def quality_control_pipeline(data, variable_name="discharge"):
    """Comprehensive quality control for time series data."""
    
    qc_results = {
        'original_length': len(data),
        'missing_count': data.isnull().sum(),
        'missing_percent': (data.isnull().sum() / len(data)) * 100
    }
    
    # 1. Range checks
    if variable_name == "discharge":
        valid_range = (0, data.quantile(0.999) * 2)  # Allow for extreme events
    elif variable_name == "temperature":
        valid_range = (-60, 60)  # Celsius
    elif variable_name == "precipitation":
        valid_range = (0, 1000)  # mm/day
    else:
        valid_range = (data.quantile(0.001), data.quantile(0.999))
    
    range_violations = (data < valid_range[0]) | (data > valid_range[1])
    qc_results['range_violations'] = range_violations.sum()
    
    # 2. Temporal consistency
    daily_changes = data.diff().abs()
    if variable_name == "discharge":
        # Flag changes > 500% of previous value
        change_threshold = data.shift(1) * 5
    else:
        # Flag changes > 3 standard deviations
        change_threshold = daily_changes.std() * 3
    
    temporal_outliers = daily_changes > change_threshold
    qc_results['temporal_outliers'] = temporal_outliers.sum()
    
    # 3. Statistical outliers
    z_scores = np.abs((data - data.mean()) / data.std())
    statistical_outliers = z_scores > 4  # Very conservative threshold
    qc_results['statistical_outliers'] = statistical_outliers.sum()
    
    # 4. Gap analysis
    time_diff = data.index.to_series().diff()
    expected_freq = pd.infer_freq(data.index)
    if expected_freq:
        expected_diff = pd.Timedelta(expected_freq)
        gaps = time_diff > expected_diff * 1.5
        qc_results['temporal_gaps'] = gaps.sum()
        qc_results['max_gap_days'] = time_diff.max().days
    
    # Create cleaned dataset
    data_clean = data.copy()
    
    # Remove range violations
    data_clean[range_violations] = np.nan
    
    # Flag but don't remove temporal outliers (might be real events)
    # data_clean[temporal_outliers] = np.nan  # Uncomment if desired
    
    qc_results['cleaned_length'] = data_clean.notna().sum()
    qc_results['data_availability'] = (data_clean.notna().sum() / len(data_clean)) * 100
    
    return data_clean, qc_results

# Example usage
discharge_clean, qc_stats = quality_control_pipeline(discharge_data, "discharge")
print(f"Data availability: {qc_stats['data_availability']:.1f}%")
print(f"Range violations: {qc_stats['range_violations']}")
```

### Automated Report Generation

```python
def generate_analysis_report(station_id, discharge_data, climate_data=None):
    """Generate comprehensive analysis report."""
    
    report = {
        'station_id': station_id,
        'analysis_date': pd.Timestamp.now(),
        'data_period': {
            'start': discharge_data.index.min(),
            'end': discharge_data.index.max(),
            'years': (discharge_data.index.max() - discharge_data.index.min()).days / 365.25
        }
    }
    
    # Quality control
    discharge_clean, qc_stats = quality_control_pipeline(discharge_data, "discharge")
    report['quality_control'] = qc_stats
    
    # Hydrological analysis
    hydro_years = split_by_hydro_year(discharge_clean)
    
    bfi_values = []
    flow_stats = []
    
    for year, data in hydro_years.items():
        if len(data) > 300:
            bfi_val, _ = bfi_1000(data.values, 3, 30)
            bfi_values.append(bfi_val)
            
            flow_stats.append({
                'year': int(year),
                'mean_flow': data.mean(),
                'min_flow': data.min(),
                'max_flow': data.max(),
                'cv': data.std() / data.mean(),
                'bfi': bfi_val
            })
    
    report['hydrological_signatures'] = {
        'bfi_mean': np.mean(bfi_values),
        'bfi_std': np.std(bfi_values),
        'bfi_trend': calculate_trend(bfi_values),  # Implement trend calculation
        'annual_stats': pd.DataFrame(flow_stats)
    }
    
    # Climate analysis (if data available)
    if climate_data is not None:
        from src.meteo.climate_indices import DroughtIndices
        drought_analyzer = DroughtIndices()
        
        spi_12 = drought_analyzer.calculate_spi(climate_data['precipitation'], scale=12)
        
        report['climate_indices'] = {
            'spi_12_mean': spi_12.mean(),
            'drought_events': (spi_12 < -1.5).sum(),
            'wet_events': (spi_12 > 1.5).sum()
        }
    
    # Generate summary
    report['summary'] = generate_summary_text(report)
    
    return report

def generate_summary_text(report):
    """Generate human-readable summary."""
    summary = f"""
    Analysis Summary for Station {report['station_id']}
    ================================================
    
    Data Period: {report['data_period']['start'].strftime('%Y-%m-%d')} to {report['data_period']['end'].strftime('%Y-%m-%d')}
    Total Years: {report['data_period']['years']:.1f}
    Data Availability: {report['quality_control']['data_availability']:.1f}%
    
    Hydrological Characteristics:
    - Mean Base Flow Index: {report['hydrological_signatures']['bfi_mean']:.3f}
    - BFI Variability: {report['hydrological_signatures']['bfi_std']:.3f}
    - Flow Regime: {"Groundwater-dominated" if report['hydrological_signatures']['bfi_mean'] > 0.5 else "Surface-flow dominated"}
    
    Quality Issues:
    - Range violations: {report['quality_control']['range_violations']}
    - Temporal outliers: {report['quality_control']['temporal_outliers']}
    """
    
    if 'climate_indices' in report:
        summary += f"""
    Climate Characteristics:
    - Drought events (SPI < -1.5): {report['climate_indices']['drought_events']}
    - Wet events (SPI > 1.5): {report['climate_indices']['wet_events']}
        """
    
    return summary
```

## Batch Processing

### Multi-Station Analysis

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def process_single_station(station_file):
    """Process a single station file."""
    try:
        # Load data
        station_id = Path(station_file).stem
        data = pd.read_csv(station_file, index_col=0, parse_dates=True)
        
        # Generate report
        report = generate_analysis_report(station_id, data['discharge'])
        
        return station_id, report
    
    except Exception as e:
        print(f"Error processing {station_file}: {e}")
        return None, None

def batch_process_stations(data_directory, output_directory):
    """Process multiple stations in parallel."""
    
    station_files = list(Path(data_directory).glob("*.csv"))
    
    # Determine number of processes
    n_processes = min(mp.cpu_count() - 1, len(station_files))
    
    results = {}
    
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        # Submit all jobs
        future_to_station = {
            executor.submit(process_single_station, file): file 
            for file in station_files
        }
        
        # Collect results
        for future in future_to_station:
            station_id, report = future.result()
            if station_id is not None:
                results[station_id] = report
                
                # Save individual report
                output_file = Path(output_directory) / f"{station_id}_report.json"
                with open(output_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
    
    print(f"Processed {len(results)} stations successfully")
    return results

# Example usage
batch_results = batch_process_stations(
    "data/discharge/", 
    "results/station_reports/"
)
```

## Visualization Workflows

### Multi-Panel Diagnostic Plots

```python
def create_diagnostic_plots(data, station_id):
    """Create comprehensive diagnostic plots."""
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'Station {station_id} - Diagnostic Analysis', fontsize=16)
    
    # 1. Time series plot
    data.plot(ax=axes[0,0])
    axes[0,0].set_title('Discharge Time Series')
    axes[0,0].set_ylabel('Discharge (m³/s)')
    
    # 2. Flow duration curve
    sorted_flows = np.sort(data.dropna())[::-1]
    exceedance_prob = np.arange(1, len(sorted_flows) + 1) / len(sorted_flows) * 100
    axes[0,1].semilogy(exceedance_prob, sorted_flows)
    axes[0,1].set_title('Flow Duration Curve')
    axes[0,1].set_xlabel('Exceedance Probability (%)')
    axes[0,1].set_ylabel('Discharge (m³/s)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Monthly climatology
    monthly_stats = data.groupby(data.index.month).agg(['mean', 'std', 'min', 'max'])
    monthly_stats['mean'].plot(ax=axes[1,0], marker='o')
    axes[1,0].fill_between(
        monthly_stats.index,
        monthly_stats['mean'] - monthly_stats['std'],
        monthly_stats['mean'] + monthly_stats['std'],
        alpha=0.3
    )
    axes[1,0].set_title('Monthly Climatology')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel('Discharge (m³/s)')
    
    # 4. Annual cycle
    daily_clim = data.groupby(data.index.dayofyear).mean()
    daily_clim.plot(ax=axes[1,1])
    axes[1,1].set_title('Annual Cycle')
    axes[1,1].set_xlabel('Day of Year')
    axes[1,1].set_ylabel('Discharge (m³/s)')
    
    # 5. Histogram
    data.hist(bins=50, ax=axes[2,0], alpha=0.7)
    axes[2,0].set_title('Discharge Distribution')
    axes[2,0].set_xlabel('Discharge (m³/s)')
    axes[2,0].set_ylabel('Frequency')
    
    # 6. Autocorrelation
    autocorr = [data.autocorr(lag=i) for i in range(50)]
    axes[2,1].plot(range(50), autocorr)
    axes[2,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[2,1].set_title('Autocorrelation Function')
    axes[2,1].set_xlabel('Lag (days)')
    axes[2,1].set_ylabel('Autocorrelation')
    
    plt.tight_layout()
    return fig

# Example usage
fig = create_diagnostic_plots(discharge_data, "12345")
fig.savefig(f"plots/station_12345_diagnostics.png", dpi=300, bbox_inches='tight')
```

## Best Practices

### Error Handling

```python
import logging
from src.utils.logger import setup_logger

logger = setup_logger("analysis_workflow")

def robust_analysis_wrapper(func, *args, **kwargs):
    """Wrapper for robust error handling."""
    try:
        return func(*args, **kwargs)
    except ValueError as e:
        logger.error(f"Value error in {func.__name__}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in {func.__name__}: {e}")
        return None

# Example usage
result = robust_analysis_wrapper(
    generate_analysis_report,
    station_id="12345",
    discharge_data=discharge_data
)
```

### Performance Optimization

```python
# Use chunked processing for large datasets
def process_large_dataset(data, chunk_size=10000):
    """Process large datasets in chunks."""
    results = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data.iloc[i:i+chunk_size]
        chunk_result = process_chunk(chunk)
        results.append(chunk_result)
    
    return pd.concat(results)

# Vectorized operations
def vectorized_bfi_calculation(discharge_matrix):
    """Calculate BFI for multiple stations simultaneously."""
    # Implementation using numpy broadcasting
    pass
```

This guide provides practical, real-world examples of how to use the CAMELS-RU analysis tools effectively for various research and operational applications.
