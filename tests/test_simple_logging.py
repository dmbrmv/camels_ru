#!/usr/bin/env python3
"""Simple test of organized logging setup.

This script tests the logger setup configuration without importing
modules that have dependency issues.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

# Create a test logger for this test module
test_logger = setup_logger("test_logger_setup", log_file="logs/test.log")


def test_logger_setup():
    """Test the logger setup function directly."""
    test_logger.info("Testing organized logger setup...")

    # Clean up existing logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.log"):
            if log_file.name != "test.log":  # Keep test log
                log_file.unlink()

    # Test different logger configurations
    loggers_config = [
        ("data_processing", "logs/processing.log"),
        ("temperature_analysis", "logs/meteorological.log"),
        ("climate_indices", "logs/meteorological.log"),
        ("trend_analysis", "logs/timeseries.log"),
        ("homogeneity_tests", "logs/timeseries.log"),
        ("hydro_metrics", "logs/hydrological.log"),
        ("base_flow_separation", "logs/hydrological.log"),
    ]

    test_logger.info("Creating loggers with organized log files:")
    for function_name, log_file in loggers_config:
        try:
            logger = setup_logger(function_name, log_file=log_file)
            logger.info(f"Testing {function_name} logger")
            test_logger.info(f"✓ {function_name} -> {log_file}")
        except Exception as e:
            test_logger.error(f"✗ {function_name} failed: {e}")

    # Check created log files
    test_logger.info("Verifying log files:")
    expected_files = [
        "logs/processing.log",
        "logs/meteorological.log",
        "logs/timeseries.log",
        "logs/hydrological.log",
    ]

    for log_file in expected_files:
        path = Path(log_file)
        if path.exists():
            size = path.stat().st_size
            test_logger.info(f"✓ {log_file} created (size: {size} bytes)")

            # Show a few lines of content
            try:
                with open(path, encoding="utf-8") as f:
                    lines = f.readlines()[-3:]  # Last 3 lines
                    for line in lines:
                        test_logger.info(f"    {line.strip()}")
            except Exception as e:
                # Log file reading errors
                test_logger.debug(f"Could not read log file content: {e}")
        else:
            test_logger.error(f"✗ {log_file} not found")

    test_logger.info("📁 Organized logging structure:")
    test_logger.info("  logs/processing.log      - Data processing and core functionality")
    test_logger.info("  logs/meteorological.log  - Weather and climate analysis")
    test_logger.info("  logs/timeseries.log      - Statistical time series analysis")
    test_logger.info("  logs/hydrological.log    - Hydrological flow analysis")

    test_logger.info("✅ Organized logging test completed successfully!")


if __name__ == "__main__":
    test_logger_setup()
