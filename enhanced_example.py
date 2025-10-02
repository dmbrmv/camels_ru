"""Example usage of enhanced CAMELS-RU analysis tools.

This script demonstrates the use of meteorological analysis, trend assessment,
and homogeneity testing tools for time series data.
"""

import numpy as np
import pandas as pd

from src.meteo.climate_indices import calculate_drought_indices
from src.meteo.temperature import calculate_temperature_metrics
from src.timeseries_stats.homogeneity import test_homogeneity
from src.timeseries_stats.trends import analyze_trends
from src.utils.logger import setup_logger

logger = setup_logger("example_analysis")


def generate_sample_data() -> tuple[pd.Series, pd.Series]:
    """Generate sample meteorological data for demonstration."""
    # Create 10 years of daily data
    dates = pd.date_range("2010-01-01", "2019-12-31", freq="D")
    n_days = len(dates)

    # Temperature with seasonal cycle and trend
    day_of_year = dates.dayofyear
    seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Add warming trend
    years = (dates.year - 2010) / 10
    trend = 2 * years  # 2°C warming over 10 years

    # Add noise
    noise = np.random.normal(0, 3, n_days)

    temperature = seasonal_temp + trend + noise

    # Precipitation with seasonal variability
    seasonal_precip = 3 + 2 * np.sin(2 * np.pi * (day_of_year - 150) / 365)
    precip_noise = np.random.exponential(seasonal_precip)
    precipitation = np.maximum(0, precip_noise)

    temp_series = pd.Series(temperature, index=dates, name="temperature")
    precip_series = pd.Series(precipitation, index=dates, name="precipitation")

    return temp_series, precip_series


def demonstrate_temperature_analysis(temp_data: pd.Series) -> None:
    """Demonstrate temperature analysis capabilities."""
    logger.info("=== Temperature Analysis ===")

    try:
        # Calculate comprehensive temperature metrics
        temp_metrics = calculate_temperature_metrics(
            temp_data,
            include_extremes=True,
            include_degree_days=True,
            cold_threshold=-10,
            hot_threshold=30,
        )

        # Display basic statistics
        basic_stats = temp_metrics["basic_statistics"]
        logger.info("Basic temperature statistics:")
        logger.info("  Mean: %.1f°C", basic_stats["mean"])
        logger.info("  Min: %.1f°C", basic_stats["min"])
        logger.info("  Max: %.1f°C", basic_stats["max"])
        logger.info("  Standard deviation: %.1f°C", basic_stats["std"])

        # Display extreme events
        extremes = temp_metrics["extreme_events"]
        logger.info("Extreme temperature events:")
        logger.info(
            "  Hot days (>30°C): %d (%.1f%%)",
            extremes["hot_days_count"],
            extremes["hot_days_percentage"],
        )
        logger.info(
            "  Cold days (<-10°C): %d (%.1f%%)",
            extremes["cold_days_count"],
            extremes["cold_days_percentage"],
        )

        # Display annual temperature trends
        annual_stats = temp_metrics["annual_statistics"]
        logger.info(
            "Annual temperature range: %.1f to %.1f°C",
            annual_stats["temperature_min"].min(),
            annual_stats["temperature_max"].max(),
        )

    except Exception as e:
        logger.error("Temperature analysis failed: %s", e)


def demonstrate_drought_analysis(precip_data: pd.Series, temp_data: pd.Series) -> None:
    """Demonstrate drought indices calculation."""
    logger.info("=== Drought Analysis ===")

    try:
        # Calculate drought indices
        drought_indices = calculate_drought_indices(
            precip_data,
            temp_data,
            include_spi=True,
            include_spei=True,
            spi_timescales=[3, 6, 12],
            spei_timescales=[6, 12],
        )

        # Display consecutive dry days statistics
        cdd = drought_indices["consecutive_dry_days"]
        logger.info("Consecutive dry days statistics:")
        logger.info("  Maximum consecutive dry days: %d", cdd["max_consecutive_dry_days"])
        logger.info("  Average consecutive dry days: %.1f", cdd["mean_consecutive_dry_days"])
        logger.info("  Total dry periods: %d", cdd["total_dry_periods"])

        # Display SPI information
        if "spi" in drought_indices:
            logger.info("SPI indices calculated for timescales: %s", list(drought_indices["spi"].keys()))

        # Display aridity index
        if "aridity_index" in drought_indices:
            aridity = drought_indices["aridity_index"]
            logger.info("Aridity index (P/PET): %.2f", aridity)

            # Classify aridity
            if aridity > 1.5:
                climate_class = "Humid"
            elif aridity > 1.0:
                climate_class = "Sub-humid"
            elif aridity > 0.5:
                climate_class = "Semi-arid"
            else:
                climate_class = "Arid"

            logger.info("Climate classification: %s", climate_class)

    except Exception as e:
        logger.error("Drought analysis failed: %s", e)


def demonstrate_trend_analysis(temp_data: pd.Series) -> None:
    """Demonstrate trend analysis capabilities."""
    logger.info("=== Trend Analysis ===")

    try:
        # Perform comprehensive trend analysis
        trend_results = analyze_trends(
            temp_data,
            variable_name="temperature",
            include_seasonal=True,
            include_annual=True,
            alpha=0.05,
        )

        # Display linear trend results
        linear_trend = trend_results["linear_trend"]
        logger.info("Linear trend analysis:")
        logger.info("  Slope: %.3f°C/year", linear_trend["slope"])
        logger.info("  Trend per decade: %.2f°C", linear_trend["trend_per_decade"])
        logger.info("  R-squared: %.3f", linear_trend["r_squared"])
        logger.info("  P-value: %.4f", linear_trend["p_value"])
        logger.info("  Trend significant: %s", linear_trend["trend_significant"])
        logger.info("  Trend direction: %s", linear_trend["trend_direction"])

        # Display Mann-Kendall results
        mk_results = trend_results["mann_kendall"]
        logger.info("Mann-Kendall test:")
        logger.info("  Trend: %s", mk_results["trend"])
        logger.info("  Z-statistic: %.3f", mk_results["z_statistic"])
        logger.info("  P-value: %.4f", mk_results["p_value"])
        logger.info("  Kendall's tau: %.3f", mk_results["tau"])

        # Display seasonal Mann-Kendall if available
        if "seasonal_mann_kendall" in trend_results:
            smk = trend_results["seasonal_mann_kendall"]
            logger.info("Seasonal Mann-Kendall test:")
            logger.info("  Trend: %s", smk["trend"])
            logger.info("  P-value: %.4f", smk["p_value"])

    except Exception as e:
        logger.error("Trend analysis failed: %s", e)


def demonstrate_homogeneity_testing(temp_data: pd.Series) -> None:
    """Demonstrate homogeneity testing capabilities."""
    logger.info("=== Homogeneity Testing ===")

    try:
        # Perform comprehensive homogeneity testing
        homogeneity_results = test_homogeneity(
            temp_data,
            variable_name="temperature",
            include_pettitt=True,
            include_buishand=True,
            include_snht=True,
            include_von_neumann=True,
            include_runs=True,
            alpha=0.05,
        )

        # Display summary
        summary = homogeneity_results["summary"]
        logger.info("Homogeneity test summary:")
        logger.info("  Inhomogeneity detected: %s", summary["inhomogeneity_detected"])
        logger.info("  Tests performed: %s", summary["tests_performed"])

        # Display individual test results
        if "pettitt_test" in homogeneity_results:
            pettitt = homogeneity_results["pettitt_test"]
            logger.info("Pettitt test:")
            logger.info("  Change point detected: %s", pettitt["change_point_detected"])
            logger.info("  P-value: %.4f", pettitt["p_value"])
            if pettitt["change_point_date"]:
                logger.info("  Change point date: %s", pettitt["change_point_date"])

        if "buishand_test" in homogeneity_results:
            buishand = homogeneity_results["buishand_test"]
            logger.info("Buishand test:")
            logger.info("  Homogeneous: %s", buishand["homogeneous"])
            logger.info("  Q-statistic: %.3f", buishand["q_statistic"])
            logger.info("  Critical value: %.3f", buishand["critical_value"])

        if "runs_test" in homogeneity_results:
            runs = homogeneity_results["runs_test"]
            logger.info("Runs test:")
            logger.info("  Random: %s", runs["random"])
            logger.info("  Runs observed: %d", runs["runs"])
            logger.info("  Runs expected: %.1f", runs["expected_runs"])
            logger.info("  P-value: %.4f", runs["p_value"])

    except Exception as e:
        logger.error("Homogeneity testing failed: %s", e)


def main() -> None:
    """Main demonstration function."""
    logger.info("Starting CAMELS-RU enhanced analysis demonstration")

    try:
        # Generate sample data
        logger.info("Generating sample meteorological data...")
        temp_data, precip_data = generate_sample_data()

        logger.info("Generated %d days of temperature and precipitation data", len(temp_data))
        logger.info("Period: %s to %s", temp_data.index[0].date(), temp_data.index[-1].date())

        # Demonstrate different analysis capabilities
        demonstrate_temperature_analysis(temp_data)
        demonstrate_drought_analysis(precip_data, temp_data)
        demonstrate_trend_analysis(temp_data)
        demonstrate_homogeneity_testing(temp_data)

        logger.info("Analysis demonstration completed successfully")

    except Exception as e:
        logger.error("Demonstration failed: %s", e)
        raise


if __name__ == "__main__":
    main()
