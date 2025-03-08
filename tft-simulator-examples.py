"""
Example scripts for running TempFlowSim with the updated Temporal Flow Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from tempflow_sim import TempFlowSim
import os
import time

# Create output directory
os.makedirs('tempflow_output', exist_ok=True)

def run_quantum_scale_simulation():
    """Run quantum scale simulation to verify interference shift prediction"""
    print("="*80)
    print("QUANTUM SCALE SIMULATION")
    print("="*80)
    
    # Configuration
    config = {
        'scale': 'quantum',
        'dimensions': 2,
        'grid_size': 128,
        'timesteps': 500,
        'save_interval': 10,
        'output_dir': 'tempflow_output',
        'random_seed': 42,
        
        # Key TFT parameters
        'eta': 5.39e-44,       # Planck time (s)
        'M_Pl': 2.18e-8,       # Planck mass (kg)
        'r_c': 8.7e-6,         # Quantum scale (m)
        'r_gal': 1e19,         # Galactic scale (m)
        'mu': 1e-4,            # Interference coupling (initial guess)
        
        # Quantum scale specific
        'quantum': {
            'dx': 1e-10,       # Grid spacing (m)
            'dt': 1e-15,       # Time step (s)
        }
    }
    
    # Create simulator
    sim = TempFlowSim(config)
    
    # Parameter tuning to match target interference shift
    print("\nStep 1: Parameter tuning to match Δφ = 2.1×10⁻⁶ rad")
    tuning_results = sim.parameter_tuning(target_delta_phi=2.1e-6, steps=100)
    
    # Run full simulation with tuned parameters
    print("\nStep 2: Running full simulation with tuned parameters")
    results = sim.run()
    
    # Analyze results
    print("\nStep 3: Analyzing results")
    analysis = sim.analyze_results(results)
    
    # Plot results
    print("\nStep 4: Creating visualization")
    sim.plot_results(results)
    
    # Save to file
    sim.save_to_file()
    
    return sim, results, analysis

def run_galactic_scale_simulation():
    """Run galactic scale simulation to examine dark matter effects"""
    print("="*80)
    print("GALACTIC SCALE SIMULATION")
    print("="*80)
    
    # Configuration
    config = {
        'scale': 'galactic',
        'dimensions': 2,
        'grid_size': 64,
        'timesteps': 100,
        'save_interval': 5,
        'output_dir': 'tempflow_output',
        'random_seed': 42,
        
        # Key TFT parameters
        'eta': 5.39e-44,       # Planck time (s)
        'M_Pl': 2.18e-8,       # Planck mass (kg)
        'r_c': 8.7e-6,         # Quantum scale (m)
        'r_gal': 1e19,         # Galactic scale (m)
        
        # Galactic scale specific
        'galactic': {
            'dx': 3.086e19,    # Grid spacing (m) - 1 kpc
            'dt': 3.154e13,    # Time step (s) - 1000 years
        }
    }
    
    # Create simulator
    sim = TempFlowSim(config)
    
    # Run simulation
    print("\nRunning galactic simulation...")
    results = sim.run()
    
    # Analyze results
    print("\nAnalyzing galactic results...")
    # Custom analysis for galactic scale
    # Calculate rotation curve from W field
    
    # Plot results
    print("\nCreating visualization...")
    sim.plot_results(results)
    
    # Save to file
    sim.save_to_file()
    
    return sim, results

def run_cosmological_scale_simulation():
    """Run cosmological scale simulation to examine Hubble parameter"""
    print("="*80)
    print("COSMOLOGICAL SCALE SIMULATION")
    print("="*80)
    
    # Configuration
    config = {
        'scale': 'cosmological',
        'dimensions': 2,
        'grid_size': 64,
        'timesteps': 100,
        'save_interval': 5,
        'output_dir': 'tempflow_output',
        'random_seed': 42,
        
        # Key TFT parameters
        'eta': 5.39e-44,       # Planck time (s)
        'M_Pl': 2.18e-8,       # Planck mass (kg)
        'r_c': 8.7e-6,         # Quantum scale (m)
        'r_gal': 1e19,         # Galactic scale (m)
        
        # Cosmological scale specific
        'cosmological': {
            'dx': 3.086e22,    # Grid spacing (m) - 1 Mpc
            'dt': 3.154e15,    # Time step (s) - 100M years
        }
    }
    
    # Create simulator
    sim = TempFlowSim(config)
    
    # Run simulation
    print("\nRunning cosmological simulation...")
    results = sim.run()
    
    # Estimate Hubble constant
    H0 = sim.estimate_hubble_constant()
    target_H0 = 70.5  # km/s/Mpc
    H0_error = abs(H0 - target_H0) / target_H0 * 100
    
    print(f"\nEstimated Hubble constant: {H0:.1f} km/s/Mpc (Target: {target_H0:.1f} km/s/Mpc)")
    print(f"Error: {H0_error:.2f}%")
    
    # Plot results
    print("\nCreating visualization...")
    sim.plot_results(results)
    
    # Save to file
    sim.save_to_file()
    
    return sim, results, H0

def run_parameter_sweep():
    """Run parameter sweep to analyze sensitivity to key parameters"""
    print("="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Base configuration for quantum scale
    base_config = {
        'scale': 'quantum',
        'dimensions': 2,
        'grid_size': 64,
        'timesteps': 200,
        'save_interval': 20,
        'output_dir': 'tempflow_output',
        'random_seed': 42,
        
        # Default parameters
        'eta': 5.39e-44,      # Planck time (s)
        'M_Pl': 2.18e-8,      # Planck mass (kg)
        'r_c': 8.7e-6,        # Quantum scale (m)
        'r_gal': 1e19,        # Galactic scale (m)
        'mu': 1e-4,           # Interference coupling
        'xi': 1e-4,           # Nonlinear coupling
        
        # Quantum scale specific
        'quantum': {
            'dx': 1e-10,      # Grid spacing (m)
            'dt': 1e-15,      # Time step (s)
        }
    }
    
    # Parameters to sweep
    params_to_sweep = {
        'mu': [0.5e-4, 1e-4, 2e-4, 4e-4],
        'xi': [0.5e-4, 1e-4, 2e-4, 4e-4],
        'r_c': [4e-6, 8.7e-6, 12e-6]
    }
    
    results = {}
    
    for param_name, param_values in params_to_sweep.items():
        print(f"\nAnalyzing sensitivity to parameter: {param_name}")
        param_results = []
        
        for value in param_values:
            # Create config with this parameter value
            config = base_config.copy()
            config[param_name] = value
            
            print(f"  Testing {param_name} = {value:.2e}")
            
            # Create simulator
            sim = TempFlowSim(config)
            
            # Run short simulation
            sim_results = sim.run()
            
            # Get final interference shift
            final_delta_phi = sim_results[-1]['interference_shift']['average']
            
            param_results.append({
                'value': value,
                'delta_phi': final_delta_phi
            })
            
            print(f"  Result: Δφ = {final_delta_phi:.3e} rad")
        
        results[param_name] = param_results
    
    # Plot parameter sensitivity
    plt.figure(figsize=(15, 5))
    
    for i, (param_name, param_results) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        
        values = [r['value'] for r in param_results]
        delta_phis = [r['delta_phi'] for r in param_results]
        
        plt.plot(values, delta_phis, 'o-')
        plt.xlabel(f"{param_name} value")
        plt.ylabel("Δφ (rad)")
        plt.title(f"Sensitivity to {param_name}")
        plt.grid(True)
        
        # Add target line
        plt.axhline(y=2.1e-6, color='r', linestyle='--', label='Target')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig("tempflow_output/parameter_sensitivity.png", dpi=300)
    
    return results

def run_multiscale_simulation():
    """Run sequential simulations at different scales"""
    # For a true multiscale simulation, we would need more sophisticated code
    # This example just runs separate simulations at different scales
    
    print("="*80)
    print("MULTI-SCALE SIMULATION")
    print("="*80)
    
    # Run quantum scale
    print("\nPhase 1: Quantum Scale")
    quantum_sim, quantum_results, _ = run_quantum_scale_simulation()
    
    # Run galactic scale
    print("\nPhase 2: Galactic Scale")
    galactic_sim, galactic_results = run_galactic_scale_simulation()
    
    # Run cosmological scale
    print("\nPhase 3: Cosmological Scale")
    cosmic_sim, cosmic_results, H0 = run_cosmological_scale_simulation()
    
    # Create summary plot
    plt.figure(figsize=(18, 6))
    
    # Plot quantum results
    plt.subplot(1, 3, 1)
    times_q = [result['time'] for result in quantum_results]
    delta_phi_q = [result['interference_shift']['average'] for result in quantum_results]
    plt.plot(times_q, delta_phi_q)
    plt.axhline(y=2.1e-6, color='r', linestyle='--', label='Target')
    plt.xlabel('Time (s)')
    plt.ylabel('Δφ (rad)')
    plt.title('Quantum Scale: Interference')
    plt.legend()
    
    # Plot galactic results
    plt.subplot(1, 3, 2)
    # Extract galactic W magnitude
    W_mag_g = [np.mean(result['W_magnitude']) for result in galactic_results]
    times_g = [result['time'] for result in galactic_results]
    plt.plot(times_g, W_mag_g)
    plt.xlabel('Time (years)')
    plt.ylabel('|W| (J/m)')
    plt.title('Galactic Scale: W Field')
    
    # Plot cosmological results
    plt.subplot(1, 3, 3)
    entropy_c = [result['entropy'] for result in cosmic_results]
    times_c = [result['time'] for result in cosmic_results]
    plt.plot(times_c, entropy_c)
    plt.xlabel('Time (Gyr)')
    plt.ylabel('Entropy (kB/m³)')
    plt.title(f'Cosmological Scale: H₀={H0:.1f} km/s/Mpc')
    
    plt.tight_layout()
    plt.savefig("tempflow_output/multiscale_summary.png", dpi=300)
    plt.close()
    
    print("\nMulti-scale simulation complete!")
    print(f"Quantum result: Δφ = {delta_phi_q[-1]:.3e} rad")
    print(f"Cosmological result: H₀ = {H0:.1f} km/s/Mpc")
    
    return {
        'quantum': quantum_results,
        'galactic': galactic_results,
        'cosmological': cosmic_results,
        'H0': H0
    }

if __name__ == "__main__":
    print("TempFlowSim Example Scripts for Updated Temporal Flow Theory")
    print("="*80)
    
    # Choose which simulation to run
    # 1: Quantum scale only
    # 2: Parameter sensitivity analysis
    # 3: Multi-scale simulation
    
    sim_choice = 1  # Change this to run different examples
    
    if sim_choice == 1:
        sim, results, analysis = run_quantum_scale_simulation()
    elif sim_choice == 2:
        sensitivity_results = run_parameter_sweep()
    elif sim_choice == 3:
        multiscale_results = run_multiscale_simulation()
    else:
        print("Invalid simulation choice")
    
    print("\nSimulation complete!")
