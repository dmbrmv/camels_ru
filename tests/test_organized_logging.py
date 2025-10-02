#!/usr/bin/env python3
"""Test organized logging to verify log files are created in the correct locations.

This script tests the new organized logging structure with separate log files
for different analysis domains.
"""

from pathlib import Path
import sys

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

# Create a test logger for this test module
test_logger = setup_logger("test_organized_logging", log_file="logs/test.log")


def test_organized_logging():
    """Test that loggers create separate log files for different domains."""
    test_logger.info("Testing organized logging structure...")

    # Clean up any existing log files for clean test
    _cleanup_log_files()

    # Test all loggers
    _test_processing_logger()
    _test_meteorological_loggers()
    _test_timeseries_loggers()
    _test_hydrological_loggers()
    _verify_log_files()

    test_logger.info("✅ Organized logging test completed!")


def _cleanup_log_files():
    """Clean up existing log files for clean test."""
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            if log_file.name != "test.log":  # Keep test log
                log_file.unlink()


def _test_processing_logger():
    """Test processing logger."""
    try:
        from processing import logger as processing_logger

        processing_logger.info("Testing data processing logger")
        test_logger.info("✓ Processing logger initialized")
    except Exception as e:
        test_logger.error(f"✗ Processing logger failed: {e}")


def _test_meteorological_loggers():
    """Test meteorological loggers."""
    try:
        from meteo.temperature import logger as temp_logger

        temp_logger.info("Testing temperature analysis logger")
        test_logger.info("✓ Temperature analysis logger initialized")
    except Exception as e:
        test_logger.error(f"✗ Temperature logger failed: {e}")

    try:
        from meteo.climate_indices import logger as climate_logger

        climate_logger.info("Testing climate indices logger")
        test_logger.info("✓ Climate indices logger initialized")
    except Exception as e:
        test_logger.error(f"✗ Climate indices logger failed: {e}")


def _test_timeseries_loggers():
    """Test time series statistics loggers."""
    try:
        from timeseries_stats.trends import logger as trends_logger

        trends_logger.info("Testing trend analysis logger")
        test_logger.info("✓ Trend analysis logger initialized")
    except Exception as e:
        test_logger.error(f"✗ Trend analysis logger failed: {e}")

    try:
        from timeseries_stats.homogeneity import logger as homog_logger

        homog_logger.info("Testing homogeneity tests logger")
        test_logger.info("✓ Homogeneity tests logger initialized")
    except Exception as e:
        test_logger.error(f"✗ Homogeneity logger failed: {e}")


def _test_hydrological_loggers():
    """Test hydrological loggers (if dependencies available)."""
    try:
        from hydro_metrics import logger as hydro_logger

        hydro_logger.info("Testing hydrological metrics logger")
        test_logger.info("✓ Hydrological metrics logger initialized")
    except Exception as e:
        test_logger.warning(f"⚠️ Hydrological logger failed (likely missing dependencies): {e}")


def _verify_log_files():
    """Check that log files were created."""
    test_logger.info("Checking log file creation:")

    expected_files = [
        "logs/processing.log",
        "logs/meteorological.log",
        "logs/timeseries.log",
        "logs/hydrological.log",
    ]

    for log_file in expected_files:
        if Path(log_file).exists():
            size = Path(log_file).stat().st_size
            test_logger.info(f"✓ {log_file} created (size: {size} bytes)")
        else:
            test_logger.error(f"✗ {log_file} not found")

    test_logger.info("📁 Log files organization:")
    test_logger.info("  logs/processing.log      - Data processing and core functionality")
    test_logger.info("  logs/meteorological.log  - Weather and climate analysis")
    test_logger.info("  logs/timeseries.log      - Statistical time series analysis")
    test_logger.info("  logs/hydrological.log    - Hydrological flow analysis")


if __name__ == "__main__":
    test_organized_logging()
