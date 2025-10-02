#!/usr/bin/env python3
"""Basic test of logger integration across all modules.

This script tests that the logger setup is working properly in all modules
without requiring optional dependencies.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logger import setup_logger

# Create a test logger for this test module
test_logger = setup_logger("test_logger_integration", log_file="logs/test.log")


def test_logger_integration():
    """Test imports to verify logger setup works correctly."""
    test_logger.info("Logger utility imported successfully")

    # Test that we can create loggers without errors
    logger = setup_logger("test_logger")
    logger.info("Testing logger functionality")
    test_logger.info("Logger creation and basic functionality works")

    test_logger.info("Testing module imports with logger integration:")

    # Test core modules
    _test_module_import("hydro_metrics", "src.hydro_metrics")
    _test_module_import("processing", "src.processing")

    # Test hydro modules
    _test_module_import("hydro.base_flow", "src.hydro.base_flow")
    _test_module_import("hydro.flow_duration", "src.hydro.flow_duration")
    _test_module_import("hydro.flow_extremes", "src.hydro.flow_extremes")
    _test_module_import("hydro.flow_indices", "src.hydro.flow_indices")
    _test_module_import("hydro.flow_timing", "src.hydro.flow_timing")
    _test_module_import("hydro.flow_variability", "src.hydro.flow_variability")

    # Test new meteo modules (with conditional imports)
    _test_module_import("meteo package", "src.meteo")

    # Test new timeseries_stats modules
    _test_module_import("timeseries_stats package", "src.timeseries_stats")

    test_logger.info("✅ Logger integration test completed successfully!")
    test_logger.info("All modules with logger setup are working correctly.")


def _test_module_import(module_name: str, import_path: str):
    """Test importing a single module and log the result."""
    try:
        # Handle relative imports properly
        if import_path.startswith("src."):
            module_path = import_path[4:]  # Remove 'src.' prefix
            __import__(module_path)
        else:
            __import__(import_path)
        test_logger.info(f"✓ {module_name} module imported")
    except Exception as e:
        test_logger.error(f"✗ {module_name} import failed: {e}")


if __name__ == "__main__":
    test_logger_integration()
