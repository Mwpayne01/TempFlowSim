"""
TempFlowSim: Temporal Flow Theory Simulation Framework
=====================================================

A numerical simulation framework for Temporal Flow Theory, modeling the dynamics of
the four-vector field W^μ derived from entanglement entropy gradients with scale-dependent
coupling across quantum, classical and cosmological scales.

Author: Matthew W. Payne
Version: 1.0.0
Date: March 5, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import laplace
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import json
from datetime import datetime

class TemporalFlowSimulation:
    """
    Main simulation class for Temporal Flow Theory.
    Implements the W^μ field dynamics across different scales.
    """
    
    def __init__(self, config=None):
        """
        Initialize the simulation with configuration parameters.
        
        Parameters:
        -----------
        config : dict or str
            Configuration dictionary or path to JSON configuration file
        """
        # Default configuration
        self.default_config = {
            'scale': 'quantum',             # 'quantum', 'classical', 'galactic', 'cosmological'
            'dimensions': 2,                # Spatial dimensions (2D or 3D)
            'grid_size': 64,                # Number of grid points in each dimension
            'timesteps': 1000,              # Number of simulation steps
            'save_interval': 10,            # Save data every N steps
            'output_dir': 'tempflow_output',# Directory to save results
            'random_seed': 42,              # Random seed for reproducibility
            
            # Physical parameters - basic set
            'coupling_eta': 6.7e-27,        # η coefficient (J·s/kg·m)
            'alpha': 1/137,                 # Fine structure constant
            'r_c': 8.7e-6,                  # Coherence scale (m)
            'r_gal': 1e19,                  # Galactic scale (m)
            'gamma_0': 1e10,                # Entropy dissipation rate (s^-1)
            'gamma_eq': 1e-20,              # Equilibrium rate (s^-1)
            
            # Scale-specific parameters
            'quantum': {
                'dx': 1e-10,                # Grid spacing (m)
                'dt': 1e-15,                # Time step (s)
                'total_time': 1e-12,        # Total simulation time (s)
                'hbar': 1.0545718e-34,      # Reduced Planck's constant (J·s)
                'mass': 9.10938356e-31      # Particle mass (electron) (kg)
            },
            'classical': {
                'dx': 1.0,                  # Grid spacing (m)
                'dt': 1e-6,                 # Time step (s)
                'total_time': 1e-3,         # Total simulation time (s)
                'G': 6.67430e-11,           # Gravitational constant (m^3/kg/s^2)
                'c': 299792458              # Speed of light (m/s)
            },
            'galactic': {
                'dx': 3.086e19,             # Grid spacing (1 kpc in m)
                'dt': 3.154e13,             # Time step (1 Myr in s)
                'total_time': 3.154e16,     # Total simulation time (1 Gyr in s)
                'v_circ': 220000            # Typical circular velocity (m/s)
            },
            'cosmological': {
                'dx': 3.086e22,             # Grid spacing (1 Mpc in m)
                'dt': 3.154e15,             # Time step (100 Myr in s)
                'total_time': 4.352e17,     # Total simulation time (13.8 Gyr in s)
                'H0': 70.0                  # Hubble constant (km/s/Mpc)
            },
            
            # Current contributions coefficients
            'sigma_q': 1.0,                # Quantum current coefficient
            'sigma_g': 0.01,               # Gravitational current coefficient
            'sigma_m': 0.1,                # Matter current coefficient
            'sigma_corr': 0.001,           # Correlation current coefficient
            
            # Field equation parameters
            'g_unified': 0.1,              # Unified coupling constant
            'V0': 4.3e-9,                  # Potential strength (J/m^3)
            'lambda': 0.1,                 # Self-interaction coefficient
            'beta': 0.01,                  # Higher-order interaction coefficient
            'delta': 0.5                   # Power-law exponent
        }
        
        # Load configuration
        if config is None:
            self.config = self.default_config
        elif isinstance(config, str):
            # Load from JSON file
            with open(config, 'r') as f:
                self.config = {**self.default_config, **json.load(f)}
        else:
            # Merge with default config
            self.config = {**self.default_config, **config}
        
        # Set random seed for reproducibility
        np.random.seed(self.config['random_seed'])
        
        # Initialize simulation parameters based on scale
        self._init_scale_parameters()
        
        # Initialize grids and fields
        self._init_grids()
        self._init_fields()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize results storage
        self.history = []
        self.current_step = 0
        
        print(f"Initialized TempFlowSim at {self.scale} scale")
        print(f"Grid size: {self.grid_size}x{self.grid_size}, dx: {self.dx} m, dt: {self.dt} s")
        print(f"Total simulation time: {self.total_time} s ({self.timesteps} steps)")
    
    def _init_scale_parameters(self):
        """Initialize parameters specific to the selected scale"""
        self.scale = self.config['scale']
        self.grid_size = self.config['grid_size']
        self.dimensions = self.config['dimensions']
        self.timesteps = self.config['timesteps']
        
        # Get scale-specific parameters
        scale_params = self.config[self.scale]
        self.dx = scale_params['dx']
        self.dt = scale_params['dt']
        self.total_time = scale_params['total_time']
        
        # Derived parameters
        if self.scale == 'quantum':
            self.hbar = scale_params['hbar']
            self.mass = scale_params['mass']
        elif self.scale == 'classical':
            self.G = scale_params['G']
            self.c = scale_params['c']
        elif self.scale == 'galactic':
            self.v_circ = scale_params['v_circ']
        elif self.scale == 'cosmological':
            self.H0 = scale_params['H0']
    
    def _init_grids(self):
        """Initialize spatial grids based on dimension"""
        if self.dimensions == 2:
            x = np.linspace(-self.grid_size//2, self.grid_size//2-1, self.grid_size) * self.dx
            y = np.linspace(-self.grid_size//2, self.grid_size//2-1, self.grid_size) * self.dx
            self.X, self.Y = np.meshgrid(x, y)
            self.R = np.sqrt(self.X**2 + self.Y**2)
            self.grid_shape = (self.grid_size, self.grid_size)
            
        elif self.dimensions == 3:
            x = np.linspace(-self.grid_size//2, self.grid_size//2-1, self.grid_size) * self.dx
            y = np.linspace(-self.grid_size//2, self.grid_size//2-1, self.grid_size) * self.dx
            z = np.linspace(-self.grid_size//2, self.grid_size//2-1, self.grid_size) * self.dx
            self.X, self.Y, self.Z = np.meshgrid(x, y, z)
            self.R = np.sqrt(self.X**2 + self.Y**2 + self.Z**2)
            self.grid_shape = (self.grid_size, self.grid_size, self.grid_size)
        
        # Precalculate coupling function across the grid
        self.coupling_g = self.g_coupling(self.R)
    
    def _init_fields(self):
        """Initialize W^μ field and matter distribution based on scale"""
        # Vector components: time + space dimensions
        vector_dims = 1 + self.dimensions
        
        if self.scale == 'quantum':
            # Gaussian wave packet for quantum scale
            sigma = 10 * self.dx
            self.rho = np.exp(-self.R**2 / (2 * sigma**2))
            self.rho = self.rho / np.sum(self.rho)  # Normalize
            
            # Initialize W field with small random fluctuations
            self.W = np.zeros((*self.grid_shape, vector_dims))
            self.W[..., 0] = 1e-10  # Small temporal component
            
            # Phase field for quantum simulations
            self.phase = np.random.uniform(0, 2*np.pi, self.grid_shape)
            
        elif self.scale == 'classical':
            # Point mass or gaussian distribution for classical scale
            sigma = 5 * self.dx
            self.rho = np.exp(-self.R**2 / (2 * sigma**2))
            central_mass = 1.0  # Normalized mass
            self.rho = self.rho * central_mass / np.sum(self.rho)
            
            # Initialize W field based on density gradient
            self.W = np.zeros((*self.grid_shape, vector_dims))
            grad_rho = np.gradient(self.rho)
            for i in range(min(len(grad_rho), self.dimensions)):
                self.W[..., i+1] = grad_rho[i] * 1e-8
            
        elif self.scale == 'galactic':
            # Exponential disk galaxy profile
            scale_length = 10 * self.dx
            self.rho = np.exp(-self.R / scale_length)
            self.rho = self.rho / np.sum(self.rho) * 1e11  # Galactic mass in solar masses
            
            # Initialize W field based on rotational structure
            self.W = np.zeros((*self.grid_shape, vector_dims))
            
            # Circular velocity field for initial conditions
            theta = np.arctan2(self.Y, self.X)
            v_phi = self.v_circ * (1 - np.exp(-self.R / scale_length))
            
            # Convert to Cartesian components
            if self.dimensions >= 2:
                self.W[..., 1] = -v_phi * np.sin(theta) / self.c**2  # x-component
                self.W[..., 2] = v_phi * np.cos(theta) / self.c**2   # y-component
            
        elif self.scale == 'cosmological':
            # Cosmic density field with small fluctuations
            mean_density = 1.0
            fluctuation = 0.01
            self.rho = mean_density + fluctuation * np.random.normal(size=self.grid_shape)
            
            # Ensure positive density
            self.rho = np.maximum(self.rho, 0.01 * mean_density)
            
            # Initialize W field for cosmological scale (Hubble flow)
            self.W = np.zeros((*self.grid_shape, vector_dims))
            self.W[..., 0] = self.H0 * 1e3 / (3e8 * 3.086e22)  # Convert km/s/Mpc to natural units
            
            # Add small divergence-free perturbations
            for i in range(1, vector_dims):
                self.W[..., i] = 1e-7 * np.random.normal(size=self.grid_shape)
    
    def g_coupling(self, r):
        """
        Scale-dependent coupling function g(r)
        
        Parameters:
        -----------
        r : ndarray
            Radial distance from origin
        
        Returns:
        --------
        ndarray
            Coupling strength at each point
        """
        r_c = self.config['r_c']
        r_gal = self.config['r_gal']
        
        # Prevent division by zero
        r_safe = np.maximum(r, 1e-10 * self.dx)
        
        # Scale function f(r)
        f_r = np.sqrt(r_safe / r_gal)
        
        # Coupling function
        g = 1.0 / (1.0 + (r_safe / (r_c * f_r))**2)
        
        return g
    
    def compute_quantum_current(self):
        """
        Compute quantum contribution to the entropy current J^μ_ent
        
        Returns:
        --------
        ndarray
            Quantum current contribution
        """
        vector_dims = 1 + self.dimensions
        J_q = np.zeros((*self.grid_shape, vector_dims))
        
        # Construct wave function from amplitude and phase
        psi = np.sqrt(self.rho) * np.exp(1j * self.phase)
        
        # Compute probability current
        grad_psi = np.gradient(psi)
        
        # Time component (proportional to rate of change of probability)
        J_q[..., 0] = np.real(1j * (np.conj(psi) * grad_psi[0] - psi * np.conj(grad_psi[0])))
        
        # Spatial components Im(ψ* ∇ψ)
        for i in range(min(len(grad_psi), self.dimensions)):
            J_q[..., i+1] = np.imag(np.conj(psi) * grad_psi[i])
        
        return self.config['sigma_q'] * self.config['hbar'] * J_q
    
    def compute_gravitational_current(self):
        """
        Compute gravitational contribution to the entropy current J^μ_ent
        
        Returns:
        --------
        ndarray
            Gravitational current contribution
        """
        vector_dims = 1 + self.dimensions
        J_g = np.zeros((*self.grid_shape, vector_dims))
        
        # Compute gravitational potential (simplified)
        potential = np.zeros(self.grid_shape)
        laplacian_rho = laplace(self.rho)
        if self.scale in ['classical', 'galactic', 'cosmological']:
            G = self.config['classical']['G']
            potential = -G * laplacian_rho
        
        # Gradient of potential
        grad_potential = np.gradient(potential)
        
        # Create current based on potential gradient and W field
        W_squared = np.sum(self.W**2, axis=-1)
        
        for i in range(min(len(grad_potential), self.dimensions)):
            J_g[..., i+1] = grad_potential[i] * W_squared
        
        return self.config['sigma_g'] * J_g
    
    def compute_matter_current(self):
        """
        Compute matter contribution to the entropy current J^μ_ent
        
        Returns:
        --------
        ndarray
            Matter current contribution
        """
        vector_dims = 1 + self.dimensions
        J_m = np.zeros((*self.grid_shape, vector_dims))
        
        # Compute matter stress-energy tensor divergence (simplified)
        div_T = np.zeros((*self.grid_shape, vector_dims))
        
        # For now, use density gradient as proxy for stress-energy divergence
        grad_rho = np.gradient(self.rho)
        for i in range(min(len(grad_rho), self.dimensions)):
            div_T[..., i+1] = grad_rho[i]
        
        return self.config['sigma_m'] * div_T
    
    def compute_correlation_current(self):
        """
        Compute correlation contribution to the entropy current J^μ_ent
        
        Returns:
        --------
        ndarray
            Correlation current contribution
        """
        # This is a simplified implementation, as the full correlation integral
        # would be computationally expensive
        vector_dims = 1 + self.dimensions
        J_corr = np.zeros((*self.grid_shape, vector_dims))
        
        # Simplified model using local density correlations
        rho_sq = self.rho**2
        grad_rho_sq = np.gradient(rho_sq)
        
        for i in range(min(len(grad_rho_sq), self.dimensions)):
            J_corr[..., i+1] = grad_rho_sq[i]
        
        return self.config['sigma_corr'] * J_corr
    
    def compute_total_current(self):
        """
        Compute the total entropy current J^μ_ent
        
        Returns:
        --------
        ndarray
            Total entropy current
        """
        J_q = self.compute_quantum_current()
        J_g = self.compute_gravitational_current()
        J_m = self.compute_matter_current()
        J_corr = self.compute_correlation_current()
        
        return J_q + J_g + J_m + J_corr
    
    def compute_potential_term(self):
        """
        Compute the potential term in the field equation
        
        Returns:
        --------
        ndarray
            Potential term for the W field update
        """
        V0 = self.config['V0']
        lambda_param = self.config['lambda']
        beta = self.config['beta']
        delta = self.config['delta']
        
        # W^2 term
        W_squared = np.sum(self.W**2, axis=-1, keepdims=True)
        
        # Compute -∂V/∂W_ν
        V_term = -V0 * (self.W + 2 * lambda_param * W_squared * self.W 
                      + beta * (2 + delta) * W_squared**(delta/2) * self.W)
        
        return V_term
    
    def update_flow_field(self):
        """
        Update the temporal flow field W^μ based on field equation
        
        Returns:
        --------
        ndarray
            Updated W field
        """
        # Compute the Laplacian of W
        laplacian_W = np.zeros_like(self.W)
        for i in range(self.W.shape[-1]):
            laplacian_W[..., i] = laplace(self.W[..., i])
        
        # Compute the coupling term W^μ ∇_μ W^ν
        advection = np.zeros_like(self.W)
        for i in range(self.W.shape[-1]):
            grad_W_i = np.gradient(self.W[..., i])
            for j in range(min(len(grad_W_i), self.W.shape[-1])):
                advection[..., i] += self.W[..., j] * grad_W_i[j]
        
        # Compute the curvature term (simplified as R^ν_μ W^μ ≈ 0 for weak fields)
        curvature_term = np.zeros_like(self.W)
        
        # Compute the potential term
        potential_term = self.compute_potential_term()
        
        # Compute the source current
        J_total = self.compute_total_current()
        
        # Update equation:
        # ∇_μ ∇^μ W^ν + g(χ) W^μ ∇_μ W^ν + R^ν_μ W^μ = -∂V/∂W_ν + g_unified J^total,ν
        W_new = self.W + self.dt * (
            laplacian_W - 
            self.coupling_g[..., np.newaxis] * advection + 
            curvature_term +
            potential_term + 
            self.config['g_unified'] * J_total
        )
        
        # Optional: Apply boundary conditions or constraints here
        
        return W_new
    
    def update_matter_distribution(self):
        """
        Update the matter distribution based on the W field
        
        Returns:
        --------
        ndarray
            Updated matter density
        """
        # Divergence of W field
        div_W = 0
        for i in range(1, self.W.shape[-1]):  # Spatial components only
            grad_W_i = np.gradient(self.W[..., i], axis=i-1)
            div_W += grad_W_i
        
        # Update equation based on continuity
        gamma_ent = self.config['gamma_0'] * (1 - self.coupling_g) + self.config['gamma_eq']
        
        rho_new = self.rho * (1 - self.dt * div_W - self.dt * gamma_ent)
        
        # Ensure positive density
        rho_new = np.maximum(rho_new, 1e-10 * np.max(rho_new))
        
        # Normalize if needed (for quantum simulations)
        if self.scale == 'quantum':
            rho_new = rho_new / np.sum(rho_new)
        
        return rho_new
    
    def update_phase(self):
        """
        Update the quantum phase field for quantum simulations
        
        Returns:
        --------
        ndarray
            Updated phase field
        """
        if self.scale != 'quantum':
            return self.phase
        
        # Simplified quantum evolution
        hbar = self.config['quantum']['hbar']
        mass = self.config['quantum']['mass']
        
        # Kinetic term from Laplacian
        laplacian_rho = laplace(self.rho)
        
        # Potential term (simplified)
        potential = np.zeros_like(self.rho)
        W_squared = np.sum(self.W**2, axis=-1)
        potential = 0.1 * W_squared
        
        # Phase evolution (~ħ∇²ρ/2m + V)
        phase_new = self.phase + self.dt * (-hbar * laplacian_rho / (2 * mass) + potential)
        
        # Wrap phase to [0, 2π]
        phase_new = phase_new % (2 * np.pi)
        
        return phase_new
    
    def step(self):
        """
        Perform a single simulation step
        
        Returns:
        --------
        bool
            True if the simulation should continue, False otherwise
        """
        # Update the W field
        W_new = self.update_flow_field()
        
        # Update the matter distribution
        rho_new = self.update_matter_distribution()
        
        # Update the phase (for quantum simulations)
        phase_new = self.update_phase()
        
        # Apply updates
        self.W = W_new
        self.rho = rho_new
        self.phase = phase_new
        
        # Save history at specified intervals
        if self.current_step % self.config['save_interval'] == 0:
            self.save_snapshot()
        
        # Increment step counter
        self.current_step += 1
        
        # Check if simulation should continue
        return self.current_step < self.timesteps
    
    def run(self):
        """
        Run the full simulation
        """
        print(f"Starting simulation with {self.timesteps} timesteps...")
        start_time = datetime.now()
        
        # Initial snapshot
        self.save_snapshot()
        
        # Main loop
        while self.step():
            # Print progress
            if self.current_step % (self.timesteps // 10) == 0:
                progress = 100 * self.current_step / self.timesteps
                elapsed = (datetime.now() - start_time).total_seconds()
                eta = elapsed * (self.timesteps - self.current_step) / self.current_step if self.current_step > 0 else 0
                print(f"Progress: {progress:.1f}% (Step {self.current_step}/{self.timesteps}), ETA: {eta:.1f}s")
        
        # Final snapshot
        self.save_snapshot()
        
        # Print summary
        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"Simulation completed in {elapsed:.2f} seconds")
        print(f"Generated {len(self.history)} snapshots")
        
        return self.history
    
    def save_snapshot(self):
        """
        Save the current state to history
        """
        # Calculate observables
        W_mag = np.sqrt(np.sum(self.W**2, axis=-1))
        
        # Energy density (simplified)
        energy_density = 0.5 * np.sum(self.W**2, axis=-1) + self.config['V0'] * np.sum(self.W**2, axis=-1)**2
        
        # Calculate total entropy (simplified)
        if self.scale == 'quantum':
            entropy = -np.sum(self.rho * np.log(np.maximum(self.rho, 1e-10)))
        else:
            entropy = np.sum(W_mag)
        
        snapshot = {
            'step': self.current_step,
            'time': self.current_step * self.dt,
            'rho': self.rho.copy(),
            'W': self.W.copy(),
            'phase': self.phase.copy() if self.scale == 'quantum' else None,
            'W_mag': W_mag,
            'energy_density': energy_density,
            'total_entropy': entropy,
            'coupling_g': self.coupling_g.copy()
        }
        
        self.history.append(snapshot)
    
    def save_results(self, filename=None):
        """
        Save simulation results to file
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save results, defaults to auto-generated name
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.config['output_dir']}/tempflow_{self.scale}_{timestamp}.npz"
        
        # Extract compact data for saving
        steps = np.array([snap['step'] for snap in self.history])
        times = np.array([snap['time'] for snap in self.history])
        rho_history = np.array([snap['rho'] for snap in self.history])
        W_mag_history = np.array([snap['W_mag'] for snap in self.history])
        energy_history = np.array([snap['energy_density'] for snap in self.history])
        entropy_history = np.array([snap['total_entropy'] for snap in self.history])
        
        # Save configuration and results
        np.savez_compressed(
            filename,
            config=json.dumps(self.config),
            steps=steps,
            times=times,
            rho=rho_history,
            W_mag=W_mag_history,
            energy=energy_history,
            entropy=entropy_history
        )
        
        print(f"Results saved to {filename}")
        
        return filename
    
    def plot_snapshot(self, snapshot_idx=-1):
        """
        Plot the state at a given snapshot index
        
        Parameters:
        -----------
        snapshot_idx : int
            Index of the snapshot to plot, defaults to latest
        """
        if not self.history:
            print("No snapshots available to plot")
            return
        
        if snapshot_idx < 0:
            snapshot_idx = len(self.history) + snapshot_idx
        
        snapshot = self.history[snapshot_idx]
        
        fig = plt.figure(figsize=(15, 10))
        
        if self.dimensions == 2:
            # Plot density
            ax1 = fig.add_subplot(221)
            im1 = ax1.imshow(snapshot['rho'], cmap='viridis', origin='lower')
            ax1.set_title(f"Density (Step {snapshot['step']})")
            plt.colorbar(im1, ax=ax1)
            
            # Plot W magnitude
            ax2 = fig.add_subplot(222)
            im2 = ax2.imshow(snapshot['W_mag'], cmap='plasma', origin='lower')
            ax2.set_title(f"W Field Magnitude")
            plt.colorbar(im2, ax=ax2)
            
            # Plot W field vectors (subsampled)
            ax3 = fig.add_subplot(223)
            subsample = max(1, self.grid_size // 20)
            ax3.quiver(
                self.X[::subsample, ::subsample], 
                self.Y[::subsample, ::subsample],
                snapshot['W'][::subsample, ::subsample, 1],
                snapshot['W'][::subsample, ::subsample, 2],
                scale=0.02
            )
            ax3.set_title(f"W Field Vectors")
            ax3.set_xlim([np.min(self.X), np.max(self.X)])
            ax3.set_ylim([np.min(self.Y), np.max(self.Y)])
            
            # Plot energy density
            ax4 = fig.add_subplot(224)
            im4 = ax4.imshow(snapshot['energy_density'], cmap='inferno', origin='lower')
            ax4.set_title(f"Energy Density")
            plt.colorbar(im4, ax=ax4)
            
        elif self.dimensions == 3:
            # For 3D, show slices through the middle
            mid_z = self.grid_size // 2
            
            # Plot density
            ax1 = fig.add_subplot(221)
            im1 = ax1.imshow(snapshot['rho'][:, :, mid_z], cmap='viridis', origin='lower')
            ax1.set_title(f"Density (z-slice, Step {snapshot['step']})")
            plt.colorbar(im1, ax=ax1)
            
            # Plot W magnitude
            ax2 = fig.add_subplot(222)
            im2 = ax2.imshow(snapshot['W_mag'][:, :, mid_z], cmap='plasma', origin='lower')
            ax2.set_title(f"W Field Magnitude (z-slice)")
            plt.colorbar(im2, ax=ax2)
            
            # Plot W field vectors (subsampled)
            ax