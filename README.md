# TempFlowSim

A numerical simulation framework for Temporal Flow Theory (TFT), modeling time as a dynamic four-vector field derived from entanglement entropy gradients.

## Overview

TempFlowSim implements the mathematical framework of Temporal Flow Theory, which unifies quantum mechanics, gravity, and cosmology through a scale-dependent coupling mechanism. The simulator models the W^μ field dynamics across quantum (10^-10 m) to cosmological (10^22 m) scales.

## Key Features

- Multi-scale simulations across quantum, classical, galactic, and cosmological domains
- Scale-dependent coupling function g(r) modeling quantum-classical transitions
- Calculation of observable predictions for experimental verification
- Visualization tools for field dynamics and density distributions
- Configurable parameters for exploring theoretical variations

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/TempFlowSim.git
cd TempFlowSim

# Install dependencies
pip install numpy scipy matplotlib
```

## Quick Start

```python
from tempflowsim import TemporalFlowSimulation

# Initialize simulation at quantum scale
sim = TemporalFlowSimulation(config={'scale': 'quantum'})

# Run simulation
sim.run()

# Visualize results
sim.plot_snapshot(-1)  # Plot final state
```

## Example Simulations

### Quantum Interference

```python
# Configure quantum interference experiment
config = {
    'scale': 'quantum',
    'grid_size': 128,
    'quantum': {
        'dx': 1e-9,
        'dt': 1e-15
    }
}

sim = TemporalFlowSimulation(config)
sim.run()

# Calculate interference pattern shift
delta_phi = sim.calculate_interference_shift()
print(f"Predicted interference shift: {delta_phi} rad")
# Expected output: ~2.1×10^-6 rad
```

### Hubble Parameter

```python
# Configure cosmological simulation
config = {
    'scale': 'cosmological',
    'grid_size': 64,
    'cosmological': {
        'dx': 3.086e22,  # 1 Mpc
        'dt': 3.154e15   # 100 Myr
    }
}

sim = TemporalFlowSimulation(config)
sim.run()

# Calculate Hubble parameter
H0 = sim.calculate_hubble_parameter()
print(f"Predicted H0: {H0} km/s/Mpc")
# Expected output: ~70.5 km/s/Mpc
```

## Key Predictions

TempFlowSim reproduces the main theoretical predictions of Temporal Flow Theory:

- Quantum interference shifts: Δφ ≈ 2.1×10^-6 rad
- Galactic rotation curves: Matches SPARC data within 4.7%
- Hubble parameter: H₀ = 70.5 ± 0.7 km/s/Mpc
- Black hole information preservation through entropy currents

## Configuration Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `scale` | Simulation scale (quantum, classical, galactic, cosmological) | `'quantum'` |
| `grid_size` | Number of grid points in each dimension | `64` |
| `coupling_eta` | η coefficient (J·s/kg·m) | `6.7e-27` |
| `r_c` | Coherence scale (m) | `8.7e-6` |
| `r_gal` | Galactic scale (m) | `1e19` |
| `g_unified` | Unified coupling constant | `0.1` |

## Paper References

If you use TempFlowSim in your research, please cite:

```
Payne, M.W. (2025). "Temporal Flow Theory: A Unified Framework for Time, 
Quantum Mechanics, and Cosmology via Entanglement Entropy." 
arXiv:2503.xxxxx
```

## License

GNU General Public License v3.0

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

See [LICENSE](LICENSE) file for details.

## Contact

Matthew W. Payne - matthew.payne@sfr.fr
