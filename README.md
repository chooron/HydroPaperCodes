# HydroPaperCodes

Code for Paper "Exploring Kolmogorov-Arnold Neural Networks for Hybrid and Transparent Hydrological Modeling"

**Notes**: new version for this project refer to https://github.com/chooron/Hydro-KAN-V1

## Project Overview

This repository contains the Julia implementation of Kolmogorov-Arnold Neural Networks (KANNs) for hydrological modeling as described in the paper. The code focuses on creating hybrid and transparent hydrological models using KANNs, with comparisons to traditional hydrological models.

## Project Structure

```
HydroPaperCodes/
├── data/                  # Input data for models
│   ├── basin_ids/         # Basin identification data
│   ├── camelsus/          # CAMELS US dataset
│   └── bucket_opt_init.csv # Initial optimization parameters
├── src/                   # Source code
│   ├── utils/             # Utility functions and helpers
│   ├── v0/                # Initial implementation version
│   ├── v1/                # First iteration of models
│   ├── v2/                # Second iteration with improved models
│   │   ├── models/        # Model definitions
│   │   ├── run_k50_base.jl # KANN model implementation
│   │   ├── run_exphydro.jl # ExpHydro model implementation
│   │   └── explain_m50_base.jl # Model explanation code
│   └── v3/                # Latest model implementations
├── result/                # Model outputs and results
│   ├── formulas/          # Generated model formulas
│   ├── models/            # Trained model parameters
│   ├── stats/             # Statistical analysis of results
│   └── v2/                # Results from v2 implementations
├── cmd/                   # Command scripts for batch processing
├── dev/                   # Development and experimental code
└── extend/                # Extensions and additional functionality
```

## Dependencies

This project relies on several Julia packages:

- `Lux`, `Flux`: Neural network frameworks
- `DifferentialEquations`, `OrdinaryDiffEq`: For solving differential equations
- `SciMLSensitivity`: Sensitivity analysis for scientific machine learning
- `HydroModels`: Hydrological modeling utilities
- `Optimization`: Optimization algorithms
- `JLD2`, `CSV`, `DataFrames`: Data handling and storage
- `Plots`, `StatsPlots`: Visualization tools
- `KolmogorovArnold`: Implementation of KANN architecture

See `Project.toml` for the complete list of dependencies.

## Usage

### Data Preparation

The code uses the CAMELS US dataset. Data should be preprocessed and stored in the `data/camelsus/` directory.

### Running Models

#### ExpHydro Model

```julia
# Run the ExpHydro model for a specific basin
include("src/v2/run_exphydro.jl")
main("10234500")  # Replace with desired basin ID
```

#### KANN Model (K50)

```julia
# Run the K50 model
include("src/v2/run_k50_base.jl")
# Model parameters are set within the script
```

### Model Explanation

```julia
# Generate explanations for model behavior
include("src/v2/explain_m50_base.jl")
```

## Results

Model outputs are stored in the `result/` directory:
- Trained model parameters in `result/models/`
- Generated formulas in `result/formulas/`
- Statistical analyses in `result/stats/`

## License

[Specify license information]

## Citation

If you use this code in your research, please cite:

```
[Citation information for the paper]
```

## Contact

[Contact information]
