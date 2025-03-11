import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import c, hbar, G, k as k_B
import scipy.optimize as optimize

class TemporalFlowTheory:
    """
    Implementation of Temporal Flow Theory (TFT) as described in the paper.
    This class provides the core functionality to compute entropy flux, g(r,T),
    and simulate various predictions of the theory.
    """
    
    def __init__(self):
        # Fundamental constants
        self.c = c  # Speed of light (m/s)
        self.hbar = hbar  # Reduced Planck constant (J·s)
        self.G = G  # Gravitational constant (m^3/kg/s^2)
        self.k_B = k_B  # Boltzmann constant (J/K)
        
        # Derived constants
        self.m_P = np.sqrt(self.hbar * self.c / self.G)  # Planck mass (kg)
        self.l_P = np.sqrt(self.hbar * self.G / self.c**3)  # Planck length (m)
        self.t_P = self.l_P / self.c  # Planck time (s)
        self.rho_P = self.m_P / self.l_P**3  # Planck density (kg/m^3)
        
        # TFT specific constants
        self.Lambda_TFT = 1.8e-52  # Cosmological constant (m^-2)
    
    def g_function(self, r, T):
        """
        Computes the g(r,T) function that governs the transition from quantum to classical regimes.
        
        Parameters:
        - r: characteristic scale (m)
        - T: temperature (K)
        
        Returns:
        - g: dimensionless weighting function
        """
        # g(r,T) = (l_P/r) * [1 - exp(-k_B*T*l_P/(hbar*c))]
        return (self.l_P / r) * (1 - np.exp(-self.k_B * T * self.l_P / (self.hbar * self.c)))
    
    def entropy_entanglement(self, r):
        """
        Computes the entanglement entropy at scale r.
        
        Parameters:
        - r: characteristic scale (m)
        
        Returns:
        - S_ent: entanglement entropy (J/K)
        """
        # S_ent = k_B * (r/l_P)^2
        return self.k_B * (r / self.l_P)**2
    
    def entropy_thermal(self, r):
        """
        Computes the thermal entropy at scale r.
        
        Parameters:
        - r: characteristic scale (m)
        
        Returns:
        - S_therm: thermal entropy (J/K)
        """
        # S_therm = k_B * V / l_P^3 where V = 4/3 * pi * r^3
        volume = (4/3) * np.pi * r**3
        return self.k_B * volume / self.l_P**3
    
    def total_entropy(self, r, T):
        """
        Computes the total entropy as a weighted sum of entanglement and thermal entropies.
        
        Parameters:
        - r: characteristic scale (m)
        - T: temperature (K)
        
        Returns:
        - S_total: total entropy (J/K)
        """
        g = self.g_function(r, T)
        S_ent = self.entropy_entanglement(r)
        S_therm = self.entropy_thermal(r)
        
        return g * S_ent + (1 - g) * S_therm
    
    def entropy_flux_W(self, r, T):
        """
        Computes the entropy flux four-vector W^μ (only time component W^0 for now).
        
        Parameters:
        - r: characteristic scale (m)
        - T: temperature (K)
        
        Returns:
        - W^0: time component of entropy flux (s^-1)
        """
        g = self.g_function(r, T)
        # W^0 ~ (c/m_P) * g * (r/l_P^2)
        return (self.c / self.m_P) * g * (r / self.l_P**2)
    
    def effective_density(self, rho, r, T):
        """
        Computes the effective density that accounts for TFT corrections.
        
        Parameters:
        - rho: matter density (kg/m^3)
        - r: characteristic scale (m)
        - T: temperature (K)
        
        Returns:
        - rho_eff: effective density (kg/m^3)
        """
        g = self.g_function(r, T)
        # ρ_eff = ρ * [1 - g * ρ/ρ_P]
        return rho * (1 - g * rho / self.rho_P)
    
    def modified_friedmann(self, a, H, rho_m, rho_r, r, T):
        """
        Computes the modified Friedmann equation with TFT corrections.
        
        Parameters:
        - a: scale factor (dimensionless)
        - H: Hubble parameter (s^-1)
        - rho_m: matter density (kg/m^3)
        - rho_r: radiation density (kg/m^3)
        - r: characteristic scale (m)
        - T: temperature (K)
        
        Returns:
        - dH/dt: time derivative of Hubble parameter (s^-2)
        """
        g = self.g_function(r, T)
        W0 = self.entropy_flux_W(r, T)
        
        # Entropy density from W^μ
        rho_W = 0.5 * self.m_P * self.c**4 * g * W0**2
        
        # Modified Friedmann equation
        H_dot = -1.5 * H**2 * (1 + (rho_r / (rho_m + rho_r + rho_W))) + 0.5 * self.Lambda_TFT * self.c**2
        
        return H_dot
    
    def bounce_simulation(self, rho_init=None, T=1e32, r_init=None, t_range=(-1e-43, 1e-43), num_points=1000):
        """
        Simulates the cosmological bounce predicted by TFT.
        
        Parameters:
        - rho_init: initial density (kg/m^3), defaults to 0.1*rho_P
        - T: temperature (K)
        - r_init: initial scale (m), defaults to l_P
        - t_range: time range to simulate (s)
        - num_points: number of time points
        
        Returns:
        - t_values: time values (s)
        - rho_values: density values (kg/m^3)
        - rho_eff_values: effective density values (kg/m^3)
        - a_values: scale factor values (dimensionless)
        """
        if rho_init is None:
            rho_init = 0.1 * self.rho_P
        
        if r_init is None:
            r_init = self.l_P
        
        # Initial conditions
        a_init = 1.0  # Normalized scale factor
        H_init = np.sqrt((8 * np.pi * self.G / 3) * rho_init)  # Initial Hubble parameter
        
        # ODE for a(t) evolution
        def ode_system(t, y):
            a, H = y
            
            # Scale r with a(t)
            r = r_init * a
            
            # Matter density scales as a^-3
            rho = rho_init / a**3
            
            # Compute effective density
            rho_eff = self.effective_density(rho, r, T)
            
            # Derivatives
            a_dot = a * H
            H_dot = -1.5 * H**2 * (1 + rho_eff / rho)
            
            return [a_dot, H_dot]
        
        # Solve ODE
        t_eval = np.linspace(t_range[0], t_range[1], num_points)
        solution = solve_ivp(ode_system, t_range, [a_init, H_init], t_eval=t_eval, method='RK45')
        
        t_values = solution.t
        a_values = solution.y[0]
        H_values = solution.y[1]
        
        # Compute densities
        rho_values = rho_init / a_values**3
        rho_eff_values = np.array([self.effective_density(rho, r_init * a, T) for rho, a in zip(rho_values, a_values)])
        
        return t_values, rho_values, rho_eff_values, a_values
    
    def nanoscale_phase_shift(self, r=1e-9, T=1.0, t_final=1.0, num_points=1000):
        """
        Computes the nanoscale quantum phase shift predicted by TFT.
        
        Parameters:
        - r: characteristic scale (m)
        - T: temperature (K)
        - t_final: final time (s)
        - num_points: number of time points
        
        Returns:
        - t_values: time values (s)
        - phase_shift: quantum phase shift (radians)
        """
        g = self.g_function(r, T)
        W0 = self.entropy_flux_W(r, T)
        
        t_values = np.linspace(0, t_final, num_points)
        
        # Phase shift from the modified Schrödinger equation
        # Δφ = ∫(ħ*c*g*W^0*exp(-g*c²/l_P*t))dt
        phase_shift = np.zeros_like(t_values)
        
        for i in range(1, len(t_values)):
            dt = t_values[i] - t_values[i-1]
            t = t_values[i-1]
            integrand = self.hbar * self.c * g * W0 * np.exp(-g * self.c**2 / self.l_P * t)
            phase_shift[i] = phase_shift[i-1] + integrand * dt
        
        return t_values, phase_shift
    
    def gravitational_wave_speed(self, r=1e23, T=2.7):
        """
        Computes the gravitational wave speed deviation predicted by TFT.
        
        Parameters:
        - r: characteristic scale (m)
        - T: temperature (K)
        
        Returns:
        - delta_c: fractional speed deviation (c_GW/c - 1)
        """
        g = self.g_function(r, T)
        W0 = self.entropy_flux_W(r, T)
        
        # c_GW/c - 1 = -(g*c⁴/m_P²)*W^μ*W_μ*exp(-g*c²/l_P*t) (at t=0)
        # We approximated W^μ*W_μ as W0²
        delta_c = -(g * self.c**4 / self.m_P**2) * W0**2
        
        return delta_c
    
    def cmb_boost(self, k_values, base_power_spectrum):
        """
        Computes the CMB power spectrum boost predicted by TFT.
        
        Parameters:
        - k_values: wavenumbers (h/Mpc)
        - base_power_spectrum: ΛCDM power spectrum values
        
        Returns:
        - boosted_spectrum: TFT-boosted power spectrum
        """
        # Constant boost of 1.0±0.5% at ℓ~100
        boost_factor = 1.01  # Mean value of 1.0%
        
        # Apply boost primarily around ℓ~100 (k~0.015 h/Mpc)
        k_peak = 0.015  # h/Mpc
        k_width = 0.008  # Width of the boost in k-space
        
        # Gaussian-like boost centered at k_peak
        k_factor = np.exp(-(k_values - k_peak)**2 / (2 * k_width**2))
        boost = 1.0 + (boost_factor - 1.0) * k_factor
        
        return base_power_spectrum * boost

    def plot_g_function(self, r_range=(1e-35, 1e26), T_values=[2.7, 1e4], figsize=(10, 6)):
        """
        Plots g(r,T) vs r for different temperatures.
        
        Parameters:
        - r_range: range of r values to plot (m)
        - T_values: list of temperature values (K)
        - figsize: figure size
        
        Returns:
        - fig, ax: matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), 1000)
        
        for T in T_values:
            g_values = np.array([self.g_function(r, T) for r in r_values])
            ax.loglog(r_values, g_values, label=f'T = {T} K')
        
        ax.set_xlabel('Scale r (m)', fontsize=12)
        ax.set_ylabel('g(r,T)', fontsize=12)
        ax.set_title('Quantum-to-Classical Transition Function g(r,T)', fontsize=14)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        return fig, ax
    
    def plot_effective_density(self, rho_range=(0, 1.5), r=None, T=1e32, figsize=(10, 6)):
        """
        Plots effective density vs density for bounce analysis.
        
        Parameters:
        - rho_range: range of density values as fraction of rho_P
        - r: characteristic scale (m), defaults to l_P
        - T: temperature (K)
        - figsize: figure size
        
        Returns:
        - fig, ax: matplotlib figure and axis objects
        """
        if r is None:
            r = self.l_P
            
        fig, ax = plt.subplots(figsize=figsize)
        
        rho_values = np.linspace(rho_range[0] * self.rho_P, rho_range[1] * self.rho_P, 1000)
        rho_eff_values = np.array([self.effective_density(rho, r, T) for rho in rho_values])
        
        # Normalize by rho_P for plotting
        rho_norm = rho_values / self.rho_P
        rho_eff_norm = rho_eff_values / self.rho_P
        
        ax.plot(rho_norm, rho_eff_norm)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.7)
        
        # Find and mark the point where rho_eff crosses zero
        zero_crossing_idx = np.where(np.diff(np.signbit(rho_eff_values)))[0]
        if len(zero_crossing_idx) > 0:
            zero_idx = zero_crossing_idx[0]
            zero_rho = rho_norm[zero_idx]
            ax.plot(zero_rho, 0, 'ro', label=f'Zero-crossing at ρ/ρ_P ≈ {zero_rho:.3f}')
        
        ax.set_xlabel('ρ/ρ_P', fontsize=12)
        ax.set_ylabel('ρ_eff/ρ_P', fontsize=12)
        ax.set_title(f'Effective Density vs. Density (r = {r:.2e} m, T = {T:.1e} K)', fontsize=14)
        ax.grid(True)
        ax.legend()
        
        return fig, ax
    
    def plot_entropy_flux(self, r_range=(1e-35, 1e26), T=2.7, figsize=(10, 6)):
        """
        Plots entropy flux W^0 vs r.
        
        Parameters:
        - r_range: range of r values to plot (m)
        - T: temperature (K)
        - figsize: figure size
        
        Returns:
        - fig, ax: matplotlib figure and axis objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        r_values = np.logspace(np.log10(r_range[0]), np.log10(r_range[1]), 1000)
        W0_values = np.array([self.entropy_flux_W(r, T) for r in r_values])
        
        ax.loglog(r_values, W0_values)
        
        ax.set_xlabel('Scale r (m)', fontsize=12)
        ax.set_ylabel('W^0 (s^-1)', fontsize=12)
        ax.set_title(f'Entropy Flux W^0 vs. Scale (T = {T} K)', fontsize=14)
        ax.grid(True, which='both', linestyle='--', alpha=0.7)
        
        return fig, ax
