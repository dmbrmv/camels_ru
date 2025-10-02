# CAMELS-RU Documentation

Welcome to the comprehensive documentation for the CAMELS-RU project. This documentation provides detailed information about methodologies, data sources, and API usage.

## Quick Navigation

### 🔬 [Methodology](methodology/)

Detailed mathematical formulations and scientific methods:

- **[Hydrological Metrics](methodology/hydrological_metrics.md)**: Base Flow Index, flow signatures, extremes analysis
- **[Meteorological Analysis](methodology/meteorological_analysis.md)**: Climate indices, temperature analysis, drought indicators
- **[Statistical Methods](methodology/statistical_methods.md)**: Time series analysis, trend detection, extreme value analysis

### 📊 [API Reference](api/)

Complete programming interface documentation:

- **[Complete Reference](api/complete_reference.md)**: Full API documentation for all modules
- Function signatures, parameters, and examples
- Error handling and performance notes

### 🌐 [Data Sources](data_sources/)

Data acquisition and processing systems:

- **[Data Retrieval Systems](data_sources/data_retrieval_systems.md)**: Automated data acquisition pipelines
- Source descriptions and quality control procedures

### 📖 [Usage Guide](usage_guide.md)

Practical examples and workflows for real-world applications

## Project Overview

CAMELS-RU is a comprehensive hydrological dataset for Russia that provides:

- **Daily discharge observations** from Russian gauge stations
- **Meteorological forcing data** including temperature, precipitation, and derived indices
- **Catchment attributes** from topography, land cover, soil, and geological sources
- **Hydrological signatures** calculated using scientifically rigorous methods

## Key Features

### Hydrological Analysis

- **Base Flow Index (BFI)**: Eckhardt digital filter with Monte Carlo uncertainty estimation
- **Flow Signatures**: 50+ metrics including timing, magnitude, duration, and variability
- **Extreme Events**: High/low flow detection and characterization
- **Flow Duration Curves**: Percentile-based flow analysis

### Meteorological Analysis

- **Climate Indices**: SPI, SPEI, PDSI for drought monitoring
- **Temperature Extremes**: Heat waves, cold spells, frost analysis
- **Seasonal Statistics**: Long-term climatology and anomaly detection
- **Degree Days**: Growing, heating, and cooling degree day calculations

### Data Processing

- **Quality Control**: Comprehensive validation and gap detection
- **Time Series Analysis**: Trend detection and statistical assessment
- **Geospatial Integration**: Catchment-based attribute extraction
- **Automated Workflows**: Reproducible processing pipelines

## Getting Started

### Basic Usage Example

```python
import pandas as pd
from src.hydro_metrics import hydro_job
from src.processing import split_by_hydro_year
from src.meteo.climate_indices import DroughtIndices

# Load and process discharge data
discharge = pd.read_csv("discharge.csv", index_col=0, parse_dates=True)
hydro_years = split_by_hydro_year(discharge['flow'])

# Calculate hydrological metrics
for year, data in hydro_years.items():
    if len(data) > 300:  # Ensure sufficient data
        hydro_job(data, data)

# Climate index analysis
precipitation = pd.read_csv("precipitation.csv", index_col=0, parse_dates=True)
drought_analyzer = DroughtIndices()
spi_3 = drought_analyzer.calculate_spi(precipitation['precip'], scale=3)
```

### Installation Requirements

```bash
# Core dependencies
pip install numpy>=1.24.0 pandas>=2.0.0 scipy>=1.10.0
pip install numba>=0.57.0 matplotlib>=3.7.0

# Geospatial analysis
pip install geopandas rasterio xarray

# Optional: Enhanced performance
pip install dask[complete]
```

## Documentation Structure

```
docs/
├── methodology/              # Scientific methods and algorithms
│   ├── hydrological_metrics.md
│   ├── meteorological_analysis.md
│   └── statistical_methods.md
├── api/                     # Programming interface
│   └── complete_reference.md
├── data_sources/            # Data acquisition systems
│   └── data_retrieval_systems.md
├── usage_guide.md          # Practical examples and workflows
└── index.md                # This file
```

## Scientific Background

The CAMELS-RU project builds upon established methodologies from the international CAMELS community:

### Base Flow Separation

Implementation follows Eckhardt (2005) with:
- Three-pass digital filtering
- Monte Carlo uncertainty quantification
- Edge effect mitigation through reflection

### Climate Indices

Standardized calculations following:

- WMO guidelines for drought indices
- Vicente-Serrano et al. (2010) for SPEI
- McKee et al. (1993) for SPI

### Hydrological Signatures

Based on Addor et al. (2018) framework with:

- Physically meaningful metrics
- Robust statistical foundations
- Climate change sensitivity

## Quality Assurance

### Code Quality

- **Type Hints**: Full type annotation for all functions
- **Linting**: Ruff-compliant code following PEP 8
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Detailed docstrings and examples

### Scientific Validation

- **Literature Comparison**: Cross-validation with published studies
- **Software Comparison**: Verification against established tools
- **Expert Review**: Scientific methodology validation
- **Reproducibility**: Fixed random seeds and documented algorithms

## Contributing

### Development Guidelines

1. **Code Standards**: Follow PEP 8 and project-specific style guidelines
2. **Testing**: Include tests for all new functionality
3. **Documentation**: Update relevant documentation files
4. **Scientific Rigor**: Provide references for new methodologies

### Pull Request Process

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request with clear description

## Support and Contact

### Issues and Questions

- **GitHub Issues**: Technical problems and feature requests
- **Documentation**: Check existing documentation first
- **Email**: [Contact information for scientific questions]

### Citation

When using CAMELS-RU in research, please cite:

```
[Citation information will be updated upon publication]
```

## Acknowledgments

This project acknowledges the foundational work of:

- CAMELS US (Newman et al., 2015)
- CAMELS GB (Addor et al., 2017)
- Hydrological signatures framework (Addor et al., 2018)
- Russian hydrometeorological community

## License

This project is licensed under [specify license] - see the LICENSE file for details.

---

**Last Updated**: [Current date]  
**Version**: [Project version]  
**Maintainers**: [Maintainer information]