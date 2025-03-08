def analyze_results(self, results=None):
        """Analyze simulation results and compare with predictions"""
        if results is None and not self.history:
            print("No results to analyze")
            return {}
        
        if results is None:
            # Extract data from history
            delta_phi_avg = [entry['observables']['interference_shift']['average'] for entry in self.history]
            delta_phi_center = [entry['observables']['interference_shift']['center'] for entry in self.history]
            times = [entry['time'] for entry in self.history]
        else:
            # Extract from results
            delta_phi_avg = [result['interference_shift']['average'] for result in results]
            delta_phi_center = [result['interference_shift']['center'] for result in results]
            times = [result['time'] for result in results]
        
        # Target values
        target_delta_phi = 2.1e-6  # rad
        target_H0 = 70.5  # km/s/Mpc
        
        # Calculate final values (average of last few steps for stability)
        num_avg_steps = min(10, len(delta_phi_avg))
        final_delta_phi = np.mean(delta_phi_avg[-num_avg_steps:])
        
        # Calculate error percentages
        delta_phi_error = abs(final_delta_phi - target_delta_phi) / target_delta_phi * 100
        
        # Hubble constant (if applicable)
        H0 = self.estimate_hubble_constant()
        H0_error = abs(H0 - target_H0) / target_H0 * 100 if H0 is not None else None
        
        # Stability metrics
        stability = {
            'delta_phi_std': np.std(delta_phi_avg[-num_avg_steps:]) / final_delta_phi * 100,
            'convergence_rate': (delta_phi_avg[-1] - delta_phi_avg[0]) / (times[-1] - times[0]) if times[-1] > times[0] else 0
        }
        
        analysis = {
            'quantum_interference': {
                'predicted': final_delta_phi,
                'target': target_delta_phi,
                'error_percent': delta_phi_error,
                'stability': stability['delta_phi_std']
            },
            'stability': stability,
            'converged': delta_phi_error < 10 and stability['delta_phi_std'] < 5
        }
        
        if H0 is not None:
            analysis['hubble_constant'] = {
                'predicted': H0,
                'target': target_H0,
                'error_percent': H0_error
            }
        
        # Print analysis
        print("\nTemporal Flow Theory Analysis:")
        print(f"Quantum Interference Shift: {final_delta_phi:.3e} rad (Target: {target_delta_phi:.1e} rad)")
        print(f"Accuracy: {100-delta_phi_error:.2f}% (Error: {delta_phi_error:.2f}%)")
        print(f"Stability: {'Good' if stability['delta_phi_std'] < 5 else 'Needs improvement'} " +
              f"- Variation: {stability['delta_phi_std']:.2f}%")
        
        if H0 is not None:
            print(f"Hubble Constant: {H0:.1f} km/s/Mpc (Target: {target_H0:.1f} km/s/Mpc)")
            print(f"Accuracy: {100-H0_error:.2f}% (Error: {H0_error:.2f}%)")
        
        return analysis
    
    def parameter_tuning(self, target_delta_phi=2.1e-6, steps=50, tune_param='mu'):
        """Auto-tune parameters to match target predictions"""
        print(f"Parameter tuning to match target Δφ = {target_delta_phi:.1e} rad")
        
        if tune_param == 'mu':
            # Initial values
            mu_values = [self.mu]
            delta_phi_values = []
            
            # Run short simulation to get current value
            original_timesteps = self.timesteps
            self.timesteps = steps
            results = self.run()
            
            # Get current delta phi
            delta_phi = results[-1]['interference_shift']['average']
            delta_phi_values.append(delta_phi)
            
            # Calculate required adjustment factor
            adjustment_factor = target_delta_phi / delta_phi
            
            # Adjust mu
            new_mu = self.mu * adjustment_factor
            print(f"Adjusting μ from {self.mu:.3e} to {new_mu:.3e} (factor: {adjustment_factor:.3f})")
            
            # Update parameter and re-run
            self.mu = new_mu
            mu_values.append(new_mu)
            
            # Reset simulation state
            self._init_fields()
            self.current_step = 0
            self.history = []
            
            # Run with new parameter
            results = self.run()
            delta_phi = results[-1]['interference_shift']['average']
            delta_phi_values.append(delta_phi)
            
            # Calculate error
            error = abs(delta_phi - target_delta_phi) / target_delta_phi * 100
            
            # Report results
            print(f"After tuning: Δφ = {delta_phi:.3e} rad (Error: {error:.2f}%)")
            
            # Restore original timesteps
            self.timesteps = original_timesteps
            
            return {
                'param': 'mu',
                'original_value': mu_values[0],
                'tuned_value': mu_values[1],
                'delta_phi_values': delta_phi_values,
                'final_error_percent': error
            }
        else:
            print(f"Tuning for parameter '{tune_param}' not implemented")
            return None

    def save_to_file(self, filename=None):
        """Save simulation results to file"""
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['output_dir']}/tempflow_{self.scale}_{timestamp}.npz"
        
        # Extract data from history
        if not self.history:
            print("No data to save")
            return
        
        # Prepare data for saving
        data = {
            'config': self.config,
            'steps': np.array([entry['step'] for entry in self.history]),
            'times': np.array([entry['time'] for entry in self.history]),
            'delta_phi_avg': np.array([entry['observables']['interference_shift']['average'] 
                                      for entry in self.history]),
            'entropy_avg': np.array([entry['observables']['entropy'] for entry in self.history])
        }
        
        # Save to file
        np.savez(filename, **data)
        print(f"Data saved to {filename}")
        
        return filename


# Main execution for demonstration
if __name__ == "__main__":
    # Example usage
    print("TempFlowSim: Python Simulator for Updated Temporal Flow Theory (TFT-2025-v1.4)")
    
    # Configuration for quantum scale simulation
    quantum_config = {
        'scale': 'quantum',
        'dimensions': 2,
        'grid_size': 64,
        'timesteps': 1000,
        'save_interval': 10,
        'output_dir': 'tempflow_output',
        'random_seed': 42,
        
        # Adjust parameters for target interference shift
        'mu': 1e-4  # Initial guess, will be tuned
    }
    
    # Create simulator
    print("\nInitializing quantum scale simulation...")
    sim = TempFlowSim(quantum_config)
    
    # Parameter tuning
    print("\nPerforming parameter tuning...")
    tuning_results = sim.parameter_tuning(target_delta_phi=2.1e-6, steps=100)
    
    # Run full simulation with tuned parameters
    print("\nRunning full simulation with tuned parameters...")
    results = sim.run()
    
    # Analyze results
    analysis = sim.analyze_results(results)
    
    # Plot results
    print("\nPlotting results...")
    sim.plot_results(results)
    
    # Save to file
    sim.save_to_file()
    
    print("\nSimulation complete!")
    
    # Example for cosmological scale (optional)
    if False:  # Set to True to run cosmological simulation
        print("\n" + "="*50)
        print("\nInitializing cosmological scale simulation...")
        cosmic_config = {
            'scale': 'cosmological',
            'dimensions': 2,
            'grid_size': 64,
            'timesteps': 100,
            'save_interval': 5,
            'output_dir': 'tempflow_output'
        }
        
        cosmic_sim = TempFlowSim(cosmic_config)
        cosmic_results = cosmic_sim.run()
        cosmic_analysis = cosmic_sim.analyze_results(cosmic_results)
        cosmic_sim.plot_results(cosmic_results)
        
        # Estimate Hubble constant
        H0 = cosmic_sim.estimate_hubble_constant()
        print(f"\nEstimated Hubble constant: {H0:.1f} km/s/Mpc (Target: 70.5 km/s/Mpc)")
