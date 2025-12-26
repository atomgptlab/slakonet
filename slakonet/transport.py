"""
SlakoNet Quantum Transport Calculator
======================================
Complete NEGF-based transport calculator for SlakoNet.
Follows GPAW/TranSIESTA methodology with proper Fermi alignment.

Features:
- Transmission spectrum calculation
- I-V curve calculation
- Support for semiconductors and metals
- Publication-quality plotting

Author: Kamal Choudhary (NIST/JHU)
Date: 2024
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union
from slakonet.main import SimpleDftb
from slakonet.atoms import Geometry
from scipy.constants import e, h, k
from scipy.integrate import simpson


class SlakoNetTransportCalculator:
    """
    Quantum transport calculator using NEGF formalism.
    
    This calculator implements the Landauer-Büttiker approach for
    quantum transport through nanoscale systems.
    
    Key Features:
    -------------
    - Separate treatment of scattering region and semi-infinite leads
    - Proper Fermi level alignment
    - Support for both semiconductors and metals
    - Integration with ASE transport framework
    
    Example Usage:
    --------------
    >>> from ase.build import bulk
    >>> from slakonet.optim import MultiElementSkfParameterOptimizer
    >>> 
    >>> # Load model
    >>> model = MultiElementSkfParameterOptimizer.load_ultra_compact("Si_only.pt")
    >>> 
    >>> # Create geometry
    >>> unit = bulk("Si", "diamond", a=5.43)
    >>> scat_atoms = unit.repeat((7, 1, 1))
    >>> lead_atoms = unit.repeat((2, 1, 1))
    >>> 
    >>> # Initialize calculator
    >>> calc = SlakoNetTransportCalculator(
    ...     model=model,
    ...     scat_atoms=scat_atoms,
    ...     lead_atoms=lead_atoms,
    ...     scat_kpts=(1, 1, 1),
    ...     lead_kpts=(4, 4, 4),
    ... )
    >>> 
    >>> # Calculate transport
    >>> energies = np.arange(-3, 3, 0.02)
    >>> T_e = calc.calculate_transmission(energies)
    >>> V, I = calc.calculate_iv_curve(energies, T_e)
    >>> calc.plot_all(energies, T_e, V, I)
    """
    
    def __init__(
        self,
        model,
        scat_atoms,
        lead_atoms,
        scat_kpts=(1, 1, 1),
        lead_kpts=(4, 4, 4),
        eta=1e-3,
        temperature=300.0,
        align_method='common',
    ):
        """
        Initialize quantum transport calculator.
        
        Parameters:
        -----------
        model : MultiElementSkfParameterOptimizer
            Trained SlakoNet model
        scat_atoms : ase.Atoms
            Scattering region (Left-Device-Right structure)
        lead_atoms : ase.Atoms
            Principal layer of semi-infinite lead
        scat_kpts : tuple of int
            k-point mesh for scattering region (kx, ky, kz)
        lead_kpts : tuple of int
            k-point mesh for leads (typically denser)
        eta : float
            Broadening parameter for Green's functions (eV)
        temperature : float
            Temperature for Fermi-Dirac distribution (K)
        align_method : str
            'common': align both to scattering E_F
            'separate': align each to its own E_F (may cause issues)
        """
        self.model = model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).float().eval()
        self.eta = eta
        self.temperature = temperature
        self.align_method = align_method
        
        self._print_header()
        
        # Calculate Hamiltonians for both regions
        self._setup_hamiltonians(scat_atoms, lead_atoms, scat_kpts, lead_kpts)
        
        # Align Fermi levels
        self._align_fermi_levels()
        
        self._print_summary()
    
    def _print_header(self):
        """Print calculator header."""
        print("\n" + "="*70)
        print("SLAKONET QUANTUM TRANSPORT CALCULATOR")
        print("Non-Equilibrium Green's Function (NEGF) Method")
        print("="*70)
    
    def _setup_hamiltonians(self, scat_atoms, lead_atoms, scat_kpts, lead_kpts):
        """Calculate Hamiltonians and overlaps for scattering and leads."""
        print("\n1. Calculating scattering region...")
        self.H_scat, self.S_scat, self.E_F_scat, self.gap_scat = \
            self._calculate_region(scat_atoms, scat_kpts, "Scattering")
        
        print("\n2. Calculating lead principal layer...")
        self.H_lead, self.S_lead, self.E_F_lead, self.gap_lead = \
            self._calculate_region(lead_atoms, lead_kpts, "Lead")
        
        # Store original E_F values
        self.E_F_scat_orig = self.E_F_scat
        self.E_F_lead_orig = self.E_F_lead
    
    def _calculate_region(self, atoms, kpts, region_name):
        """
        Calculate electronic structure for a region.
        
        Returns H, S, E_F, band_gap
        """
        geometry = Geometry.from_ase_atoms([atoms])
        geometry.positions.requires_grad_(False)
        
        calc = SimpleDftb(
            geometry,
            kpoints=torch.tensor(kpts),
            model=self.model,
            compute_forces=False,
            include_dos_data=False
        )
        calc.calculate()
        
        # Extract matrices
        H = calc.hamiltonian[..., 0].squeeze()
        S = calc.overlap[..., 0].squeeze()
        
        # Convert to real (handle complex from k-points)
        if torch.is_complex(H):
            H = H.real
        if torch.is_complex(S):
            S = S.real
        
        E_F = calc.fermi_energy.item()
        gap = calc.bandgap.item()
        
        # Print info
        print(f"   {region_name}:")
        print(f"     Atoms:      {len(atoms)}")
        print(f"     k-points:   {kpts}")
        print(f"     Orbitals:   {H.shape[0]}")
        print(f"     E_F:        {E_F:.4f} eV")
        print(f"     Band gap:   {gap:.4f} eV")
        
        return H, S, E_F, gap
    
    def _align_fermi_levels(self):
        """
        Align Fermi levels using specified method.
        
        The alignment H' = H - E_F * S shifts the energy scale
        so that E_F becomes the new zero of energy.
        """
        print(f"\n3. Fermi level alignment (method: {self.align_method})...")
        print(f"   Scattering E_F: {self.E_F_scat:.4f} eV")
        print(f"   Lead E_F:       {self.E_F_lead:.4f} eV")
        
        E_diff = abs(self.E_F_scat - self.E_F_lead)
        print(f"   Difference:     {E_diff:.4f} eV")
        
        if E_diff > 0.1:
            print(f"   ⚠️  WARNING: Large Fermi level mismatch!")
            print(f"      This may indicate incompatible electronic structures.")
            print(f"      Consider using same k-points or checking geometry.")
        
        if self.align_method == 'common':
            # Align both to scattering region Fermi level
            E_F_ref = self.E_F_scat
            print(f"   Using common reference: {E_F_ref:.4f} eV")
            
            self.H_scat = self.H_scat - E_F_ref * self.S_scat
            self.H_lead = self.H_lead - E_F_ref * self.S_lead
            
            # Store the reference
            self.E_F_reference = E_F_ref
            
        elif self.align_method == 'separate':
            # Align each to its own Fermi level (may cause issues)
            print(f"   ⚠️  Using separate alignment (not recommended)")
            
            self.H_scat = self.H_scat - self.E_F_scat * self.S_scat
            self.H_lead = self.H_lead - self.E_F_lead * self.S_lead
            
            self.E_F_reference = 0.0  # Ambiguous in this case
        else:
            raise ValueError(f"Unknown align_method: {self.align_method}")
    
    def _print_summary(self):
        """Print setup summary."""
        print(f"\n4. System summary:")
        print(f"   Scattering: {self.H_scat.shape[0]} orbitals")
        print(f"   Lead:       {self.H_lead.shape[0]} orbitals")
        print(f"   Temperature: {self.temperature} K")
        print(f"   Broadening η: {self.eta} eV")
        print("="*70 + "\n")
    
    def calculate_transmission(
        self,
        energies: np.ndarray,
        use_ase: bool = True,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Calculate transmission spectrum T(E).
        
        Uses the Fisher-Lee relation:
        T(E) = Trace[Γ_L G^R Γ_R G^A]
        
        where:
        - G^R is the retarded Green's function
        - G^A is the advanced Green's function
        - Γ_L,R are the broadening matrices from the leads
        
        Parameters:
        -----------
        energies : np.ndarray
            Energy points to evaluate (eV, relative to aligned E_F)
        use_ase : bool
            Use ASE's transport calculator (recommended)
        verbose : bool
            Print progress information
        
        Returns:
        --------
        T_e : np.ndarray
            Transmission coefficient at each energy
        """
        if not use_ase:
            raise NotImplementedError(
                "Manual NEGF implementation not yet available. "
                "Please use use_ase=True (default)."
            )
        
        try:
            from ase.transport.calculators import TransportCalculator
        except ImportError:
            raise ImportError(
                "ASE transport module required!\n"
                "Install with: pip install ase\n"
                "or: conda install -c conda-forge ase"
            )
        
        if verbose:
            print("Calculating transmission spectrum...")
            print(f"  Energy range: {energies[0]:.2f} to {energies[-1]:.2f} eV")
            print(f"  Number of points: {len(energies)}")
        
        # Convert to numpy complex (ASE requirement)
        h = self.H_scat.detach().cpu().numpy().astype(complex)
        s = self.S_scat.detach().cpu().numpy().astype(complex)
        h1 = self.H_lead.detach().cpu().numpy().astype(complex)
        s1 = self.S_lead.detach().cpu().numpy().astype(complex)
        
        # Initialize ASE transport calculator
        tcalc = TransportCalculator(
            h=h,        # Scattering Hamiltonian
            h1=h1,      # Left lead Hamiltonian
            h2=h1,      # Right lead (assume symmetric)
            s=s,        # Scattering overlap
            s1=s1,      # Left lead overlap
            s2=s1,      # Right lead overlap
            align_bf=1, # Align Fermi levels
            eta=self.eta,
        )
        
        # Calculate transmission
        tcalc.set(energies=energies)
        T_e = tcalc.get_transmission()
        
        if verbose:
            idx_zero = np.argmin(np.abs(energies))
            print(f"✓ Calculation complete")
            print(f"  T(E=0):  {T_e[idx_zero]:.4f}")
            print(f"  T_max:   {np.max(T_e):.4f} at E = {energies[np.argmax(T_e)]:.2f} eV")
            print(f"  T_min:   {np.min(T_e):.4f}")
        
        return T_e
    
    def calculate_iv_curve(
        self,
        energies: np.ndarray,
        T_e: np.ndarray,
        voltages: Optional[np.ndarray] = None,
        temperature: Optional[float] = None,
        verbose: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate I-V characteristic using Landauer-Büttiker formula.
        
        The current is given by:
        I(V) = (2e²/h) ∫ T(E)[f_L(E-μ_L) - f_R(E-μ_R)] dE
        
        where μ_L = +eV/2 and μ_R = -eV/2 for symmetric bias drop.
        
        Parameters:
        -----------
        energies : np.ndarray
            Energy grid (same as used for T_e)
        T_e : np.ndarray
            Transmission spectrum
        voltages : np.ndarray, optional
            Voltage points (V). Default: -2 to +2 V, 41 points
        temperature : float, optional
            Temperature (K). Default: use self.temperature
        verbose : bool
            Print progress
        
        Returns:
        --------
        voltages : np.ndarray
            Voltage points (V)
        currents : np.ndarray
            Current at each voltage (A)
        """
        if voltages is None:
            voltages = np.linspace(-2.0, 2.0, 41)
        
        if temperature is None:
            temperature = self.temperature
        
        if verbose:
            print(f"\nCalculating I-V curve...")
            print(f"  Voltage range: {voltages[0]:.2f} to {voltages[-1]:.2f} V")
            print(f"  Temperature: {temperature} K")
        
        def fermi_dirac(E, mu, T):
            """Fermi-Dirac distribution function."""
            kT = k * T / e  # Convert to eV
            arg = np.clip((E - mu) / kT, -50, 50)  # Prevent overflow
            return 1.0 / (1.0 + np.exp(arg))
        
        currents = []
        G0 = 2 * e**2 / h  # Quantum conductance (7.748e-5 S)
        
        for V in voltages:
            # Chemical potentials (symmetric bias drop)
            mu_L = +V / 2
            mu_R = -V / 2
            
            # Fermi window = difference in occupations
            f_L = fermi_dirac(energies, mu_L, temperature)
            f_R = fermi_dirac(energies, mu_R, temperature)
            
            # Landauer integral
            integrand = T_e * (f_L - f_R)
            I = simpson(integrand, x=energies) * G0
            
            currents.append(I)
        
        currents = np.array(currents)
        
        if verbose:
            idx_zero = np.argmin(np.abs(voltages))
            dV = voltages[1] - voltages[0]
            G = np.gradient(currents, dV)
            
            print(f"✓ Calculation complete")
            print(f"  I(V=0):  {currents[idx_zero]*1e9:.3f} nA")
            print(f"  I(+2V):  {currents[-1]*1e6:.3f} μA")
            print(f"  I(-2V):  {currents[0]*1e6:.3f} μA")
            print(f"  G(V=0):  {G[idx_zero]*1e6:.3f} μS")
        
        return voltages, currents
    
    def plot_all(
        self,
        energies: np.ndarray,
        T_e: np.ndarray,
        voltages: Optional[np.ndarray] = None,
        currents: Optional[np.ndarray] = None,
        save_path: str = "transport_results.png",
        show: bool = True,
    ):
        """
        Create publication-quality plot of all results.
        
        Parameters:
        -----------
        energies : np.ndarray
            Energy points
        T_e : np.ndarray
            Transmission spectrum
        voltages : np.ndarray, optional
            Voltage points
        currents : np.ndarray, optional
            Current values
        save_path : str
            Path to save figure
        show : bool
            Display figure interactively
        """
        if voltages is not None and currents is not None:
            # Full plot with T(E), I-V, and dI/dV
            fig = plt.figure(figsize=(15, 5))
            gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)
            
            # Panel 1: Transmission spectrum
            ax1 = fig.add_subplot(gs[0])
            ax1.plot(energies, T_e, linewidth=2, color='#1f77b4')
            ax1.axvline(0, color='k', linestyle='--', 
                       alpha=0.3, linewidth=1, label='E_F')
            ax1.set_xlabel('Energy (eV)', fontsize=11)
            ax1.set_ylabel('Transmission T(E)', fontsize=11)
            ax1.set_title('Transmission Spectrum', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.legend(fontsize=9)
            
            # Add gap shading if semiconductor
            if self.gap_scat > 0.01:
                gap_half = self.gap_scat / 2
                ax1.axvspan(-gap_half, gap_half, alpha=0.1, 
                           color='gray', label=f'Gap ({self.gap_scat:.2f} eV)')
            
            # Panel 2: I-V characteristic
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(voltages, currents*1e6, 'o-', linewidth=2, 
                    markersize=4, color='#ff7f0e')
            ax2.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax2.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax2.set_xlabel('Voltage (V)', fontsize=11)
            ax2.set_ylabel('Current (μA)', fontsize=11)
            ax2.set_title('I-V Characteristic', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3, linestyle=':')
            
            # Panel 3: Differential conductance
            ax3 = fig.add_subplot(gs[2])
            dV = voltages[1] - voltages[0]
            G = np.gradient(currents, dV)
            ax3.plot(voltages, G*1e6, 'o-', linewidth=2,
                    markersize=4, color='#d62728')
            ax3.axhline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax3.axvline(0, color='k', linestyle='--', alpha=0.3, linewidth=1)
            ax3.set_xlabel('Voltage (V)', fontsize=11)
            ax3.set_ylabel('dI/dV (μS)', fontsize=11)
            ax3.set_title('Differential Conductance', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, linestyle=':')
            
        else:
            # Transmission only
            fig, ax1 = plt.subplots(figsize=(8, 6))
            ax1.plot(energies, T_e, linewidth=2.5, color='#1f77b4')
            ax1.axvline(0, color='k', linestyle='--', 
                       alpha=0.3, linewidth=1.5, label='E_F')
            ax1.set_xlabel('Energy (eV)', fontsize=13)
            ax1.set_ylabel('Transmission T(E)', fontsize=13)
            ax1.set_title('Transmission Spectrum', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, linestyle=':')
            ax1.legend(fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


def silicon_nanowire_example():
    """
    Example: Silicon nanowire transport calculation.
    
    Demonstrates semiconductor transport with:
    - Band gap blocking at small bias
    - Nonlinear I-V characteristic
    - Threshold voltage behavior
    """
    from ase.build import bulk
    from slakonet.optim import MultiElementSkfParameterOptimizer
    
    print("\n" + "="*70)
    print("EXAMPLE 1: SILICON NANOWIRE (SEMICONDUCTOR)")
    print("="*70)
    
    # Load model
    model = MultiElementSkfParameterOptimizer.load_ultra_compact("Si_only.pt")
    
    # Create Si wire structure
    unit_cell = bulk("Si", "diamond", a=5.43)
    scat_atoms = unit_cell.repeat((7, 1, 1))  # 7 unit cells
    lead_atoms = unit_cell.repeat((2, 1, 1))  # 2 unit cells
    
    print(f"\nGeometry:")
    print(f"  Scattering: {len(scat_atoms)} Si atoms (7 unit cells)")
    print(f"  Lead:       {len(lead_atoms)} Si atoms (2 unit cells)")
    
    # Initialize calculator
    calc = SlakoNetTransportCalculator(
        model=model,
        scat_atoms=scat_atoms,
        lead_atoms=lead_atoms,
        scat_kpts=(4,4,4),   # Gamma point for scattering
        lead_kpts=(4, 4, 4),   # Dense k-mesh for leads
        eta=1e-3,
        temperature=300.0,
        align_method='common',
    )
    
    # Calculate transmission spectrum
    energies = np.arange(-3, 3, 0.02)
    T_e = calc.calculate_transmission(energies)
    
    # Calculate I-V curve
    voltages = np.linspace(-2.0, 2.0, 41)
    voltages, currents = calc.calculate_iv_curve(energies, T_e, voltages)
    
    # Plot results
    calc.plot_all(
        energies, T_e, voltages, currents,
        save_path="si_nanowire_transport.png"
    )
    
    # Print summary
    print("\n" + "="*70)
    print("SILICON NANOWIRE RESULTS:")
    print("="*70)
    print(f"  Band gap:        {calc.gap_scat:.3f} eV")
    print(f"  T(E_F):          {T_e[np.argmin(np.abs(energies))]:.4f}")
    print(f"  T_max:           {np.max(T_e):.4f}")
    print(f"  Threshold V:     ~{voltages[np.where(np.abs(currents) > 1e-7)[0][0]]:.2f} V")
    print(f"  I @ ±2V:         ±{abs(currents[-1])*1e6:.2f} μA")
    print("="*70 + "\n")
    
    return calc, energies, T_e, voltages, currents


def aluminum_nanowire_example():
    """
    Example: Aluminum nanowire transport calculation.
    
    Demonstrates metallic transport with:
    - Finite conductance at E_F
    - Linear Ohmic I-V curve
    - Constant differential conductance
    """
    from ase.build import bulk
    from slakonet.optim import MultiElementSkfParameterOptimizer
    
    print("\n" + "="*70)
    print("EXAMPLE 2: ALUMINUM NANOWIRE (METAL)")
    print("="*70)
    
    # Load model
    model = MultiElementSkfParameterOptimizer.load_ultra_compact("Al_only.pt")
    
    # Create Al wire structure
    unit_cell = bulk("Al", "fcc", a=4.05)
    scat_atoms = unit_cell.repeat((6, 1, 1))  # 6 unit cells (even number)
    lead_atoms = unit_cell.repeat((2, 1, 1))  # 2 unit cells
    
    print(f"\nGeometry:")
    print(f"  Scattering: {len(scat_atoms)} Al atoms (6 unit cells)")
    print(f"  Lead:       {len(lead_atoms)} Al atoms (2 unit cells)")
    
    # Initialize calculator
    # Key: Use same k-points for better compatibility
    calc = SlakoNetTransportCalculator(
        model=model,
        scat_atoms=scat_atoms,
        lead_atoms=lead_atoms,
        scat_kpts=(6,6,6),   # Same k-points as lead
        lead_kpts=(6,6,6),   # For better Fermi alignment
        eta=0.05,              # Larger broadening for metal
        temperature=300.0,
        align_method='common',
    )
    
    # Calculate transmission spectrum
    energies = np.arange(-5, 5, 0.02)  # Wider range for metal
    T_e = calc.calculate_transmission(energies)
    
    # Calculate I-V curve
    voltages = np.linspace(-2.0, 2.0, 41)
    voltages, currents = calc.calculate_iv_curve(energies, T_e, voltages)
    
    # Plot results
    calc.plot_all(
        energies, T_e, voltages, currents,
        save_path="al_nanowire_transport.png"
    )
    
    # Print summary
    print("\n" + "="*70)
    print("ALUMINUM NANOWIRE RESULTS:")
    print("="*70)
    print(f"  Band gap:        {calc.gap_scat:.3f} eV (metallic)")
    print(f"  T(E_F):          {T_e[np.argmin(np.abs(energies))]:.4f}")
    print(f"  T_max:           {np.max(T_e):.4f}")
    idx_zero = np.argmin(np.abs(voltages))
    G_zero = np.gradient(currents, voltages[1]-voltages[0])[idx_zero]
    print(f"  G(V=0):          {G_zero*1e6:.2f} μS")
    print(f"  I @ ±2V:         ±{abs(currents[-1])*1e6:.2f} μA")
    print(f"  I-V behavior:    {'Linear (Ohmic)' if np.std(np.gradient(currents)) < 1e-8 else 'Nonlinear'}")
    print("="*70 + "\n")
    
    return calc, energies, T_e, voltages, currents


def compare_si_al():
    """
    Compare semiconductor vs metal transport side-by-side.
    """
    print("\n" + "="*70)
    print("COMPARATIVE STUDY: SEMICONDUCTOR vs METAL TRANSPORT")
    print("="*70 + "\n")
    
    # Run both examples
    print("Running Silicon example...")
    si_calc, si_E, si_T, si_V, si_I = silicon_nanowire_example()
    
    print("\n" + "-"*70 + "\n")
    
    print("Running Aluminum example...")
    al_calc, al_E, al_T, al_V, al_I = aluminum_nanowire_example()
    
    # Create comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Silicon
    axes[0, 0].plot(si_E, si_T, 'b-', linewidth=2)
    axes[0, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_title('Si: Transmission', fontweight='bold')
    axes[0, 0].set_xlabel('Energy (eV)')
    axes[0, 0].set_ylabel('T(E)')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(si_V, si_I*1e6, 'b-', linewidth=2, marker='o', markersize=3)
    axes[0, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_title('Si: I-V Curve', fontweight='bold')
    axes[0, 1].set_xlabel('Voltage (V)')
    axes[0, 1].set_ylabel('Current (μA)')
    axes[0, 1].grid(True, alpha=0.3)
    
    si_G = np.gradient(si_I, si_V[1]-si_V[0])
    axes[0, 2].plot(si_V, si_G*1e6, 'b-', linewidth=2, marker='o', markersize=3)
    axes[0, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_title('Si: Conductance', fontweight='bold')
    axes[0, 2].set_xlabel('Voltage (V)')
    axes[0, 2].set_ylabel('dI/dV (μS)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Aluminum
    axes[1, 0].plot(al_E, al_T, 'r-', linewidth=2)
    axes[1, 0].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_title('Al: Transmission', fontweight='bold')
    axes[1, 0].set_xlabel('Energy (eV)')
    axes[1, 0].set_ylabel('T(E)')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(al_V, al_I*1e6, 'r-', linewidth=2, marker='o', markersize=3)
    axes[1, 1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_title('Al: I-V Curve', fontweight='bold')
    axes[1, 1].set_xlabel('Voltage (V)')
    axes[1, 1].set_ylabel('Current (μA)')
    axes[1, 1].grid(True, alpha=0.3)
    
    al_G = np.gradient(al_I, al_V[1]-al_V[0])
    axes[1, 2].plot(al_V, al_G*1e6, 'r-', linewidth=2, marker='o', markersize=3)
    axes[1, 2].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].set_title('Al: Conductance', fontweight='bold')
    axes[1, 2].set_xlabel('Voltage (V)')
    axes[1, 2].set_ylabel('dI/dV (μS)')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('si_vs_al_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to si_vs_al_comparison.png")
    plt.show()
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON TABLE:")
    print("="*70)
    print(f"{'Property':<30} {'Silicon':<20} {'Aluminum':<20}")
    print("-"*70)
    print(f"{'Band gap (eV)':<30} {si_calc.gap_scat:<20.3f} {al_calc.gap_scat:<20.3f}")
    print(f"{'T(E_F)':<30} {si_T[np.argmin(np.abs(si_E))]:<20.4f} {al_T[np.argmin(np.abs(al_E))]:<20.4f}")
    print(f"{'Max T(E)':<30} {np.max(si_T):<20.4f} {np.max(al_T):<20.4f}")
    print(f"{'I @ 2V (μA)':<30} {si_I[-1]*1e6:<20.3f} {al_I[-1]*1e6:<20.3f}")
    print(f"{'G @ V=0 (μS)':<30} {si_G[len(si_V)//2]*1e6:<20.3f} {al_G[len(al_V)//2]*1e6:<20.3f}")
    print(f"{'I-V character':<30} {'Nonlinear':<20} {'Linear (Ohmic)':<20}")
    print("="*70 + "\n")



    #     # Run comparison by default
    #     compare_si_al()
