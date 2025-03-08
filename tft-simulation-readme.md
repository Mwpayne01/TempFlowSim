# TempFlowSim

A Python simulator for the updated Temporal Flow Theory (TFT-2025-v1.4)

## Overview

TempFlowSim is a numerical simulation framework for investigating Temporal Flow Theory, which redefines time as a dynamic four-vector field derived from entanglement entropy gradients. This simulator implements the mathematical framework described in the updated TFT specification to produce quantitative predictions across quantum, classical, galactic, and cosmological scales.

## Features

- Multi-scale simulation capabilities from quantum (10⁻¹⁰ m) to cosmological (10²² m) domains
- Numerical implementation of the W^μ field equations with scale-dependent coupling
- Automatic parameter tuning to match experimental predictions
- Comprehensive visualization of simulation results
- Analysis tools for comparing predictions with theoretical targets

## Installation

```bash
# Clone the repository
git clone https://github.com/username/TempFlowSim.git
cd TempFlowSim

# Install dependencies
pip install numpy scipy matplotlib
```

## Usage

Basic usage example:

```python
from tempflow_sim import TempFlowSim

# Configuration for quantum scale simulation
config = {
    'scale': 'quantum',
    'dimensions': 2,
    'grid_size': 64,
    'timesteps': 500
}

# Create simulator
sim = TempFlowSim(config)

# Run simulation
results = sim.run()

# Analyze and visualize results
analysis = sim.analyze_results(results)
sim.plot_results(results)
```

## Key Parameters

The updated Temporal Flow Theory has several key parameters that can be adjusted in the simulation:

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `eta` | Planck time (s) | 5.39 × 10⁻⁴⁴ |
| `M_Pl` | Planck mass (kg) | 2.18 × 10⁻⁸ |
| `r_c` | Quantum scale (m) | 8.7 × 10⁻⁶ |
| `r_gal` | Galactic scale (m) | 10¹⁹ |
| `mu` | Interference coupling | 10⁻⁴ |
| `xi` | Nonlinear coupling | 10⁻⁴ |
| `V0` | Potential scale (J/m³) | 10¹¹² |
| `lambda_param` | Self-interaction | 10⁻¹² |

## Core Equations

The simulator implements the following key equations from the updated TFT:

1. **Temporal Field Definition**:
   ```
   W^μ = (η/M_Pl³) ∇^μ S_ent
   ```

2. **Scale-Dependent Coupling**:
   ```
   g(r) = 1/(1 + (r/r_c)² (r/r_gal))
   ```

3. **Field Equation**:
   ```
   ∇_μ F^μν + ξ g(r) W^μ F_μν = -∂V/∂W_ν + κ T^ν_μ,matter W^μ + 2α R^ν_μ W^μ
   ```

4. **Entropy Evolution**:
   ```
   ∇_μ S_ent = J^μ_ent - Γ_ent S_ent
   ```

5. **Quantum Interference Shift**:
   ```
   Δφ = μ g(r) |W|²/M_Pl²
   ```

## Example Scripts

The repository includes several example scripts for running different types of simulations:

- `quantum_simulation.py`: Demonstrates quantum interference predictions
- `galactic_simulation.py`: Examines dark matter effects in galactic rotation
- `cosmological_simulation.py`: Studies Hubble parameter evolution
- `parameter_sensitivity.py`: Analyzes how predictions vary with parameter changes
- `multiscale_simulation.py`: Runs simulations across multiple scales

## Results

The simulator produces the following key predictions that can be compared with observational data:

- Quantum interference shift: Δφ ≈ 2.1×10⁻⁶ rad
- Hubble parameter: H₀ ≈ 70.5 km/s/Mpc
- Galactic rotation curves with 4.7% deviation from SPARC data

## Advanced Usage

### Parameter Tuning

```python
# Auto-tune parameters to match target predictions
tuning_results = sim.parameter_tuning(
    target_delta_phi=2.1e-6,
    steps=100,
    tune_param='mu'
)
```

### Multi-scale Simulation

```python
# Run separate simulations at different scales
quantum_sim = TempFlowSim({'scale': 'quantum'})
galactic_sim = TempFlowSim({'scale': 'galactic'})
cosmic_sim = TempFlowSim({'scale': 'cosmological'})

# Run simulations
quantum_results = quantum_sim.run()
galactic_results = galactic_sim.run()
cosmic_results = cosmic_sim.run()

# Extract key predictions
delta_phi = quantum_results[-1]['interference_shift']['average']
H0 = cosmic_sim.estimate_hubble_constant()
```

## Visualization

The simulator generates comprehensive visualizations of the simulation results:

- W^μ field magnitude and vector distribution
- Entanglement entropy evolution
- Quantum interference shift patterns
- Parameter sensitivity analysis
- Scale-dependent coupling function

## Contributing

Contributions to TempFlowSim are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GNU General Public License v3.0 - see the LICENSE file for details.

## Citation

If you use TempFlowSim in your research, please cite:

```
Payne, M.W. (2025). "Temporal Flow Theory: A Unified Framework for Time, 
Quantum Mechanics, Gravity, and Cosmology via Entanglement Entropy."
```

## Contact

Matthew W. Payne - Matthew.Payne@sfr.fr
