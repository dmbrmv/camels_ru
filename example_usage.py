"""Example usage of the hydro metrics package.

This script demonstrates how to use the reorganized hydrological metrics
modules for comprehensive discharge analysis.
"""

import pandas as pd

# Import the hydro modules
from src.hydro import (
    BaseFlowSeparation,
    FlowDurationCurve,
    FlowExtremes,
    FlowTiming,
    FlowVariability,
    HydrologicalIndices,
    calculate_comprehensive_metrics,
)


def analyze_discharge_series(discharge: pd.Series) -> dict[str, float]:
    """Comprehensive analysis of discharge time series.

    Args:
        discharge: Discharge time series with datetime index

    Returns:
        Dictionary of comprehensive hydrological metrics
    """
    print("Starting comprehensive discharge analysis...")

    # 1. Base Flow Analysis
    print("1. Calculating base flow separation...")
    base_flow_sep = BaseFlowSeparation()
    bfi, base_flow_series = base_flow_sep.separate(discharge)
    base_flow_stats = base_flow_sep.calculate_baseflow_stats(discharge)

    # 2. Flow Duration Curve Analysis
    print("2. Analyzing flow duration curves...")
    fdc = FlowDurationCurve(discharge)
    fdc_metrics = {
        "fdc_slope": fdc.calculate_fdc_slope(),
        **fdc.calculate_flow_percentiles(),
        **fdc.calculate_variability_indices(),
    }

    # 3. Flow Extremes Analysis
    print("3. Analyzing flow extremes...")
    extremes = FlowExtremes(discharge)
    extreme_metrics = {
        **extremes.analyze_high_flows(),
        **extremes.analyze_low_flows(),
        **extremes.calculate_drought_indices(),
        **extremes.calculate_flood_indices(),
    }

    # 4. Flow Timing Analysis
    print("4. Analyzing flow timing...")
    if isinstance(discharge.index, pd.DatetimeIndex):
        timing = FlowTiming(discharge)
        timing_metrics = {
            **timing.calculate_half_flow_date(),
            **timing.calculate_seasonal_flows(),
            **timing.calculate_extreme_timing(),
            **timing.calculate_flow_duration_metrics(),
        }
    else:
        timing_metrics = {}

    # 5. Flow Variability Analysis
    print("5. Analyzing flow variability...")
    variability = FlowVariability(discharge)
    variability_metrics = {
        **variability.calculate_basic_statistics(),
        **variability.calculate_temporal_variability(),
        **variability.calculate_autocorrelation_metrics(),
        **variability.calculate_flashiness_index(),
    }

    # 6. Comprehensive Hydrological Indices
    print("6. Calculating hydrological indices...")
    indices = HydrologicalIndices(discharge)
    hydrological_indices = {
        **indices.calculate_magnitude_indices(),
        **indices.calculate_frequency_indices(),
        **indices.calculate_duration_indices(),
        **indices.calculate_timing_indices(),
        **indices.calculate_rate_of_change_indices(),
    }

    # Combine all metrics
    all_metrics = {
        **{"baseflow_" + k: v for k, v in base_flow_stats.items()},
        **{"fdc_" + k: v for k, v in fdc_metrics.items()},
        **{"extreme_" + k: v for k, v in extreme_metrics.items()},
        **{"timing_" + k: v for k, v in timing_metrics.items()},
        **{"variability_" + k: v for k, v in variability_metrics.items()},
        **{"index_" + k: v for k, v in hydrological_indices.items()},
    }

    print(f"Analysis complete. Calculated {len(all_metrics)} metrics.")
    return all_metrics


def demonstrate_quick_analysis(discharge: pd.Series) -> dict[str, float]:
    """Quick comprehensive analysis using the convenience function.

    Args:
        discharge: Discharge time series

    Returns:
        Dictionary of metrics
    """
    print("Running quick comprehensive analysis...")
    return calculate_comprehensive_metrics(discharge, include_bfi=True, include_all_modules=True)


if __name__ == "__main__":
    # Example with synthetic data
    import numpy as np

    # Create synthetic discharge data
    dates = pd.date_range("2010-01-01", "2020-12-31", freq="D")

    # Generate synthetic discharge with seasonal pattern and noise
    seasonal_component = 10 + 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.lognormal(0, 0.5, len(dates))
    synthetic_discharge = seasonal_component * noise

    discharge_series = pd.Series(synthetic_discharge, index=dates)

    # Run comprehensive analysis
    metrics = analyze_discharge_series(discharge_series)

    # Display some key metrics
    print("\nKey Hydrological Metrics:")
    print(f"BFI: {metrics.get('baseflow_bfi', 'N/A'):.3f}")
    print(f"FDC Slope: {metrics.get('fdc_fdc_slope', 'N/A'):.3f}")
    print(f"CV: {metrics.get('variability_cv', 'N/A'):.3f}")
    print(f"Flashiness Index: {metrics.get('variability_flashiness_index', 'N/A'):.3f}")

    # Quick analysis demonstration
    quick_metrics = demonstrate_quick_analysis(discharge_series)
    print(f"\nQuick analysis calculated {len(quick_metrics)} metrics.")
