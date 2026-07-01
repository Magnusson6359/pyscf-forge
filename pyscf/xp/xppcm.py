'''
XPPCM module for PySCF.
'''

# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------
# XP, XPCav: main classes for XP-PCM and XP-PCM with cavitation energy
#   - E_rep available via mf.with_xp.get_e_rep()
#   - E_cav available via mf.with_xp.get_e_cav() (XPCav only)
#   - XPCav gradients available via mf.with_xp.nuc_grad_method()
#   - Built via factory functions ks.XP(...) and ks.XPCav(...), which wrap standard SCF objects
# get_z0, get_z, get_eps: helpers for calculating XP-PCM parameters from solvent properties
# calculate_fitted_pressure: fit E(V) to an equation of state and compute pressure
# calculate_numerical_pressure: compute pressure by numerical differentiation of E(V)
# calculate_analytical_pressure: compute pressure analytically 
# get_cavity_volume_grid: compute cavity volume via grid integration (numerical, slow)
# get_cavity_volume_atom: compute cavity volume for a single atom (analytical, fastest)
# get_cavity_volume: compute cavity volume (analytical, fast)

import numpy as np
import warnings
from scipy.optimize import curve_fit
from pyscf import lib, dft, gto, ci, cc
from pyscf.data import radii
from pyscf.dft.gen_grid import MakeAngularGrid, LEBEDEV_ORDER

# AU pressure conversion factor: 1 Eh/Bohr^3 = 29421.015798 GPa
EH_BOHR3_TO_GPA = 29421.015798

Rahm = radii.RAHM.copy()
Vdw  = radii.VDW.copy()
Mod_Bondi = Vdw.copy()
Mod_Bondi[1] = 1.1 / radii.BOHR

# ---------------------------------------------------------------------------
#  Public helpers
# ---------------------------------------------------------------------------

def get_z0(solvent_num_valence_elec, solvent_density, solvent_molar_mass):
    """
    Calculate the initial barrier height z_0 from solvent parameters.

    Parameters
    ----------
    solvent_num_valence_elec : int
        Number of valence electrons in a solvent molecule.
    solvent_density : float
        Density of the solvent (g/cm^3).
    solvent_molar_mass : float
        Molar mass of the solvent (g/mol).

    Returns
    -------
    float
        Semi-empirical barrier height / implicit electron density in atomic units.
    """
    return (solvent_num_valence_elec * solvent_density / solvent_molar_mass) * 0.063

def get_z(f_vdw, f_vdw_0, z0, eta=6):
    """
    Calculate the barrier height z at a given cavity scaling factor.

    Parameters
    ----------
    f_vdw : float
        Current scaling factor.
    f_vdw_0 : float
        Reference (equilibrium) scaling factor.
    z0 : float
        Barrier height at f_vdw_0.
    eta : int, default 6
        Repulsive exponent.

    Returns
    -------
    float
        Semi-empirical barrier height / implicit electron density in atomic units.
    """
    return z0 * (f_vdw / f_vdw_0) ** -(3 + eta)

def get_eps(f_vdw, f_vdw_0, eps_0):
    """
    Calculate the dielectric constant at a given cavity scaling factor.
    Theoretically sound for non-polar solvents.

    Parameters
    ----------
    f_vdw : float
        Current scaling factor.
    f_vdw_0 : float
        Reference scaling factor.
    eps_0 : float
        Dielectric constant at f_vdw_0.

    Returns
    -------
    float
        Dielectric constant.
    """
    return 1 + (eps_0 - 1) * (f_vdw / f_vdw_0) ** -3

def calculate_fitted_pressure(energies, volumes):
    """
    Calculate pressure from a fitted Birch–Murnaghan-type equation of state.
    Cannot capture electronic transitions but works well for smooth E(V) curves.

    Parameters
    ----------
    energies : array
        Total energies (Hartree).
    volumes : array
        Cavity volumes (Bohr^3, atomic units).

    Returns
    -------
    array
        Pressures (GPa) at each volume.
    """
    def _eos(x, a, b, c):
        return (a / b) * (1 / x) ** b + (a - c) * x

    volumes = np.asarray(volumes)
    energies = np.asarray(energies)

    x_data = volumes / volumes[0]
    y_data = energies - energies[0]

    popt, _ = curve_fit(_eos, x_data, y_data)
    a_fit, b_fit, c_fit = popt

    a = a_fit / volumes[0]
    b = b_fit
    c = c_fit / volumes[0]
    
    v_ratio = volumes[0] / volumes
    return (a * (v_ratio ** (b + 1) - 1) + c) * EH_BOHR3_TO_GPA

def calculate_numerical_pressure(energies, volumes):
    """
    Calculate pressure by numerical differentiation of E(V).
    If repulsion energy is included, a high number of radial grid points is strongly recommended.
    (at least 1000 radial grid points for smooth curves)

    Parameters
    ----------
    energies : array
        Energies (Hartree). Can be total energies or separate contribtions (i.e. polarisation energy from PCM).
    volumes : array
        Cavity volumes (Bohr^3, atomic units).

    Returns
    -------
    array
        Pressures (GPa) at each volume.
    """
    energies = np.asarray(energies, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    pressure = np.zeros_like(energies)

    # Central difference for interior points
    for i in range(1, len(energies) - 1):
        dE_dV = (energies[i + 1] - energies[i - 1]) / (volumes[i + 1] - volumes[i - 1])
        pressure[i] = dE_dV * EH_BOHR3_TO_GPA

    # Forward / backward difference for endpoints
    pressure[0]  = (energies[1] - energies[0]) / (volumes[1] - volumes[0]) * EH_BOHR3_TO_GPA
    pressure[-1] = (energies[-1] - energies[-2]) / (volumes[-1] - volumes[-2]) * EH_BOHR3_TO_GPA

    return -pressure

def calculate_analytical_pressure(method_obj, volume_cavity, eta=6, lebedev_order=29):
    """
    Calculate pressure from the XP-PCM SCF analytical expression.
    Cammi, R., Chen, B. & Rahm, M., 
    Journal of Computational Chemistry 39, 2243–2250 (2018).
    Does not include polarisation effects from PCM.

    Parameters
    ----------
    method_obj : method object
        either scf object with XP attached, or a post-HF object (e.g. CI, CC) with an _SCF attribute that has XP attached.
    volume_cavity : float
        Cavity volume (Bohr^3, atomic units).
    eta : float, default 6
        Repulsive exponent.
    lebedev_order : int, default 29
        Lebedev quadrature order for surface grids (29 = 302 points per sphere).
    grid_level : int, default 3
        Grid level for integration.
    radial_grid_pts : int or None, default 1000
        Number of radial grid points for integration. 
        None = default PySCF radial grid for the given level.

    Returns
    -------
    float
        Pressure in GPa.
    """
    # Check if pcm is attached. If so, warn that the analytical pressure will not include polarisation effects.
    if hasattr(method_obj, 'with_solvent'):
        warnings.warn(
            """
            Analytical pressure calculation does not include polarisation effects from PCM.
            If wished, it can be included in post via calculate_numerical_pressure by supplying 
            the polarisation energy via .with_solvent.e.
            """,
            UserWarning,
            stacklevel=2
        )
    
    xp = method_obj.with_xp
    z = xp.z
    radii_table = xp.radii_table
    mol = method_obj.mol

    post_hf = hasattr(method_obj, '_scf') # Post HF objects have an _scf attribute that contains the SCF object.
    if post_hf:
        dm_tot = _get_post_hf_dm_tot(method_obj)
    else: # SCF method with XP attached directly
        dm = method_obj.make_rdm1()
        dm_tot = dm[0] + dm[1] if np.shape(dm)[0] == 2 else dm
    
    # Term 1: Repulsive energy contribution
    e_rep = method_obj.with_xp.get_e_rep(dm_tot, mol=mol)
    term1 = ((3.0 + eta) / (3.0 * volume_cavity)) * e_rep

    # Term 2: Surface electron density contribution
    surface_data = _get_cavity_surface_grid(
        mol, radii_table=radii_table,
        lebedev_order=lebedev_order)

    numerator_surfint = 0.0   # Σ_i Σ_exposed(w_l * ρ_l)
    denominator_surfint = 0.0 # Σ_i Σ_exposed(w_l)

    for atom in range(mol.natm):
        sd = surface_data[atom]
        w = sd['weights']
        
        if len(sd['coords']) > 0:
            # Evaluate electron density ρ(r) = Σ_μν P_μν φ_μ(r) φ_ν(r)
            ao_surf = dft.numint.eval_ao(mol, sd['coords'], deriv=0)
            rho_surf = np.einsum('gi,ij,gj->g', ao_surf, dm_tot, ao_surf)
            
            # Σ_exposed(w_l * ρ_l)
            numerator_surfint += np.sum(w * rho_surf)
            
            # Σ_exposed(w_l)
            denominator_surfint += np.sum(w)

    # Compute weighted surface density
    if denominator_surfint > 1e-14:
        term2 = z * numerator_surfint / denominator_surfint
    else:
        term2 = 0.0

    p_gpa = (term1 + term2) * EH_BOHR3_TO_GPA

    return p_gpa

def get_cavity_volume_grid(mol, radii_table, grid_level=3, radial_grid_pts=2000):
    """
    Volume of the van der Waals cavity, computed via grid integration.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Table of van der Waals radii for each atomic species.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).
    radial_grid_pts : int or None, default 2000
        Number of radial grid points. If None, use standard PySCF radial grid for the given level.

    Returns
    -------
    float
        Volume in Bohr^3.
    """
    coords, weights = _build_grid(mol, grid_level, radial_grid_pts=radial_grid_pts)
    atom_coords = mol.atom_coords()
    radii_list = _get_cavity_radii(mol, radii_table=radii_table)

    diff = coords[None, :, :] - atom_coords[:, None, :]    # (natm, ngrid, 3)
    dist2 = np.einsum('aig,aig->ai', diff, diff)           # (natm, ngrid)
    radii2 = np.array(radii_list)[:, None] ** 2
    inside_any = (dist2 <= radii2).any(axis=0)              # (ngrid,)
    vol = np.sum(weights[inside_any])
    return vol

def get_cavity_volume_atom(mol, radii_table):
    """
    Exact volume of the cavity for a single atom.
    Raises ``ValueError`` for multi-atom molecules.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object. Must contain exactly one atom.
    radii_table : array
        Table of van der Waals radii for each atomic species.

    Returns
    -------
    float
        Volume in Bohr^3.
    """
    if mol.natm != 1:
        raise ValueError("get_cavity_volume_atom is only valid for single-atom molecules.")
    R = _get_cavity_radii(mol, radii_table=radii_table)[0]
    return (4.0 / 3.0) * np.pi * R ** 3

from .analytical_cavity import get_cavity_volume # Analytical cavity volume

# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _get_cavity_radii(mol, radii_table):
    """
    Resolve cavity radii in Bohr as a per-atom array.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Table of van der Waals radii for each atomic species.

    Returns
    -------
    array
        Array of cavity radii in Bohr.
    """
    return np.array([
        radii_table[gto.charge(mol.atom_symbol(i))]
        for i in range(mol.natm)
    ], dtype=float)

def _build_grid(mol, grid_level, radial_grid_pts=1000):
    """
    Build a PySCF grid for evaluation of XP-PCM integrals.
    Note: radial grid increase per default only acts on the XP-PCM integrals, 
    not on the DFT part if using with a DFT functional. 
    This is to avoid unnecessarily increasing the cost of the DFT part, 
    while ensuring smooth XP-PCM energies and pressures. 
    If wished, the radial grid for the DFT part can be increased by setting 
    mf.grids.radial_grid = radial_grid_pts when using with a DFT functional.
    
    Parameters
    ----------
    mol : gto.Mole
    grid_level : int
        PySCF grid level for integration (default: 3).
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use standard PySCF radial grid for the given level.

    Returns
    -------
    coords : array, shape (ngrid, 3)
        Grid coordinates in Bohr.
    weights : array, shape (ngrid,)
        Grid weights.
    """
    from pyscf.dft.gen_grid import _default_ang
    
    grid = dft.gen_grid.Grids(mol)
    
    if radial_grid_pts is not None:
        # Use custom radial grid with angular points from the specified level
        atom_grid = {}
        for ia in range(mol.natm):
            symb = mol.atom_symbol(ia)
            if symb not in atom_grid:
                chg = gto.charge(symb)
                n_ang = _default_ang(chg, level=grid_level)
                atom_grid[symb] = (radial_grid_pts, n_ang)
        grid.atom_grid = atom_grid
    else:
        # Use standard PySCF grid for the specified level
        grid.level = grid_level

    grid.radi_method = dft.radi.delley
    grid.build(with_non0tab=False)
    return grid.coords.copy(), grid.weights.copy()

def _get_confined_grid(mol, radii_table, grid_level=3, radial_grid_pts=1000):
    """
    Build a grid and mask out points inside the cavity to get the confined (exterior) grid.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Table of van der Waals radii for each atomic species.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use standard PySCF radial grid for the given level.

    Returns
    -------
    coords : array, shape (ngrid, 3)
        Grid coordinates in Bohr.
    weights : array, shape (ngrid,)
        Grid weights.
    """
    coords, weights = _build_grid(mol, grid_level, radial_grid_pts=radial_grid_pts)

    atom_coords = mol.atom_coords()
    radii_list = _get_cavity_radii(mol, radii_table=radii_table)
    inside = np.zeros(len(coords), dtype=bool)
    for idx in range(mol.natm):
        R = radii_list[idx]
        dist = np.linalg.norm(coords - atom_coords[idx], axis=1)
        inside |= (dist <= R)

    weights[inside] = 0.0
    return coords, weights

def _get_S_out(mol, radii_table, grid_level=3, radial_grid_pts=1000):
    """
    Overlap matrix S_out evaluated on the confined (exterior) grid.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Table of van der Waals radii for each atomic species.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use standard PySCF radial grid for the given level.
        Per-atom van der Waals radii. Must provide one of radii_table or per_atom_radii.

    Returns
    -------
    array, shape (nao, nao)
        Overlap matrix in the AO basis outside the cavity.
    """
    coords, weights = _get_confined_grid(
        mol, radii_table=radii_table,
        grid_level=grid_level, radial_grid_pts=radial_grid_pts)
    ao = dft.numint.eval_ao(mol, coords, deriv=0)
    weighted_ao = ao * weights[:, np.newaxis]
    return ao.T @ weighted_ao

def _get_S_out_nuc_grad(mol, dm, radii_table,
                        grid_level=3, radial_grid_pts=1000):
    """
    Analytical nuclear gradient of S_out w.r.t. nuclear coordinates.

    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    dm : array, shape (nao, nao)
        Density matrix in the AO basis.
    radii_table : array
        Table of van der Waals radii for each atomic species.
    grid_level : int, default 3
        PySCF grid level for integration (default: 3).
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use standard PySCF radial grid for the given level.

    Returns
    -------
    array, shape (natm, 3)
        In atomic units (Bohr^-1).
    """
    coords, weights = _get_confined_grid(
        mol, radii_table=radii_table,
        grid_level=grid_level, radial_grid_pts=radial_grid_pts)

    ao_all = dft.numint.eval_ao(mol, coords, deriv=1)
    ao  = ao_all[0]        # (ngrid, nao)
    dao = ao_all[1:4]      # (3, ngrid, nao)

    Vphi = (ao * weights[:, np.newaxis]) @ dm   # (ngrid, nao)
    contrib = np.einsum('kgm,gm->km', dao, Vphi)  # (3, nao)

    grad = np.zeros((mol.natm, 3))
    ao_slices = mol.aoslice_by_atom()
    for A in range(mol.natm):
        p0, p1 = ao_slices[A][2], ao_slices[A][3]
        grad[A, :] = -2.0 * contrib[:, p0:p1].sum(axis=1)
    return grad

def _get_cavity_surface_grid(mol, radii_table, lebedev_order=29):
    """
    Exposed Lebedev surface grid on each atom's cavity sphere.
    Parameters
    ----------
    mol : gto.Mole
        Molecule object.
    radii_table : array
        Table of van der Waals radii for each atomic species.
    lebedev_order : int, default 29
        Order of the Lebedev grid.

    Returns
    -------
    list of dict
        One dict per atom with keys 'coords', 'normals', 'weights', 'radius'.
    """
    atom_coords = mol.atom_coords()                          # (natm, 3) Bohr
    cavity_radii = _get_cavity_radii(mol, radii_table=radii_table)

    npts  = LEBEDEV_ORDER[lebedev_order]
    grid  = MakeAngularGrid(npts)          # (m, 4): x, y, z, w  (w sums to 1)
    dirs  = grid[:, :3]
    leb_w = grid[:, 3]

    surface_data = []
    for A in range(mol.natm):
        R_A = cavity_radii[A]
        surf_pts = atom_coords[A] + R_A * dirs                # (m, 3)

        exposed = np.ones(len(dirs), dtype=bool)
        for B in range(mol.natm):
            if B == A:
                continue
            dist2 = np.sum((surf_pts - atom_coords[B])**2, axis=1)
            exposed &= dist2 > cavity_radii[B]**2

        surface_data.append({
            'coords':  surf_pts[exposed],
            'normals': dirs[exposed],
            'weights': leb_w[exposed],
            'radius':  R_A,
        })

    return surface_data

# ---------------------------------------------------------------------------
#  XP (Pauli repulsion only)
# ---------------------------------------------------------------------------

class _XP:
    """
    Encapsulates Pauli repulsion energy and Hamiltonian calculations.

    This class manages the configuration and computes two types of corrections:
    - Hamiltonian correction: h_rep = z·S_out (added to core Hamiltonian each SCF iteration)
    - Energy correction: E_rep = z·Tr[P·S_out] (for diagnostics/reporting)

    where z is a semi-empirical barrier height and S_out is the overlap matrix
    evaluated on the cavity exterior.

    Intended for internal use by _SCFWithXP; not for direct instantiation.
    Use XP() factory function or mf.XP().

    Parameters
    ----------
    mol : gto.Mole
    z : float
        Semi-empirical barrier height/Implicit electron density.
    radii_table : ndarray, optional
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    per_atom_radii : list or ndarray, optional
        Cavity radii in Bohr, indexed by atom index.
    grid_level : int, default 3
        PySCF grid level for XP overlap integrals.
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use grid_level default.
    """

    def __init__(self, mol, z, radii_table, grid_level=3, radial_grid_pts=1000):
        self.mol = mol
        self.z = z
        self.radii_table = radii_table
        self.grid_level = int(grid_level)
        self.radial_grid_pts = radial_grid_pts

    def get_h_rep(self, mol=None):
        """h_rep = z · S_out."""
        mol = mol or self.mol
        return _get_S_out(
            mol, radii_table=self.radii_table, grid_level=self.grid_level, 
            radial_grid_pts=self.radial_grid_pts) * self.z

    def get_e_rep(self, dm, mol=None):
        """E_rep = z · Tr[P · S_out].  Handles restricted and unrestricted densities."""
        mol = mol or self.mol
        dm_tot = dm[0] + dm[1] if np.shape(dm)[0] == 2 else dm
        S_out = _get_S_out(
            mol, radii_table=self.radii_table,
            grid_level=self.grid_level, radial_grid_pts=self.radial_grid_pts)
        return np.trace(dm_tot @ S_out) * self.z

class _SCFWithXP:
    """
    SCF wrapper that includes XP corrections to the Hamiltonian and energy.
    """
    _keys = {'with_xp'}

    def __init__(self, mf, xp_params):
        self.__dict__.update(mf.__dict__)
        self.with_xp = xp_params

    @property
    def radii_table(self):
        """Expose radii_table for PCM compatibility."""
        return self.with_xp.radii_table
    
    @radii_table.setter
    def radii_table(self, value):
        """Allow setting radii_table for PCM compatibility."""
        self.with_xp.radii_table = value

    def undo_xp(self):
        """Remove XP, returning the underlying SCF object."""
        cls = self.__class__
        obj = lib.view(self, lib.drop_class(cls, _SCFWithXP, 'XP'))
        del obj.with_xp
        return obj

    def get_hcore(self, mol=None):
        return super().get_hcore(mol) + self.with_xp.get_h_rep(mol)

    def get_e_rep(self, dm=None, mol=None):
        """Get Pauli repulsion energy for diagnostics."""
        if dm is None:
            dm = self.make_rdm1()
        return self.with_xp.get_e_rep(dm, mol)

    def nuc_grad_method(self):
        raise NotImplementedError(
            "Nuclear gradients are not defined for XP because "
            "geometry optimisation under XPPCM requires a cavitation energy contribution. (Otherwise cavity is fixed) "
            "Use XPCav(mf, z, radii_table, p_gpa) for geometry optimisation under a fixed pressure.")

    Gradients = nuc_grad_method

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        xp = self.with_xp
        log = lib.logger.new_logger(self, verbose)
        log.info('')
        log.info('******** XP parameters ********')
        log.info('z (barrier height)    = %.3f', xp.z)
        log.info('grid_level            = %d', xp.grid_level)
        return self

def XP(mf, z, radii_table, grid_level=3, radial_grid_pts=1000):
    """
    Attach XP (Pauli repulsion only) to an SCF method.
    
    Usage:
        ks = dft.RKS(mol)
        ks_xp = XP(ks, z=0.5, radii_table=Vdw)
        ks_xp.kernel()
    
    Parameters
    ----------
    mf : pyscf.scf.SCF
        SCF object to wrap.
    z : float
        Semi-empirical barrier height.
    radii_table : array, 
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    grid_level : int, default 3
        PySCF grid level for XP integrals.
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use grid_level default.
    
    Returns
    -------
    Modified SCF object with XP Hamiltonian correction
    """
    if isinstance(mf, _SCFWithXP):
        raise ValueError(
            "Cannot attach XP to an object that already has XP.")
    
    with_xp = _XP(
        mf.mol, z, radii_table=radii_table,
        grid_level=grid_level, radial_grid_pts=radial_grid_pts)
    
    sol_mf = _SCFWithXP(mf, with_xp)
    name = 'XP' + mf.__class__.__name__
    return lib.set_class(sol_mf, (_SCFWithXP, mf.__class__), name)

# ---------------------------------------------------------------------------
#  XPCav (Pauli repulsion + cavitation)
# ---------------------------------------------------------------------------

class _XPCav(_XP):
    """
    Encapsulates Pauli repulsion and approximate cavitation plus gradient calculations.

    Extends _XP with fixed-pressure cavitation: computes E_cav = pV in addition to
    h_rep and E_rep. Also provides analytical nuclear gradients for geometry optimisation
    under constant external pressure.

    Intended for internal use by _SCFWithXPCav; not for direct instantiation.
    Use XPCav() factory function or mf.XPCav().

    Parameters
    ----------
    mol : gto.Mole
    z : float
        Semi-empirical barrier height/Implicit electron density.
    radii_table : array
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    p_gpa : float
        External pressure (GPa).
    grid_level : int, default 3
        PySCF grid level for volume and overlap integrals.
    lebedev_order : int, default 29
        Lebedev quadrature order for surface integrals (→ 302 points).
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use grid_level default.
    """

    def __init__(self, mol, z, radii_table=None,
                 p_gpa=None, grid_level=3, lebedev_order=29, radial_grid_pts=1000):
        super().__init__(mol, z, radii_table=radii_table,
                         grid_level=grid_level, radial_grid_pts=radial_grid_pts)
        self.p_gpa = p_gpa
        self.lebedev_order = int(lebedev_order)

    def get_e_cav(self, mol=None):
        """E_cav = pV in Hartree."""
        mol = mol or self.mol
        V = get_cavity_volume(
            mol, radii_table=self.radii_table)
        return V * self.p_gpa / EH_BOHR3_TO_GPA

    def grad(self, dm):
        """Nuclear gradient of (E_rep + E_cav), shape (natm, 3), Hartree/Bohr.
        
        Computes three contributions:
        1. Cavitation:  dE_cav/dR_A = p · R_A^2 · 4π · Σ_l w_l n_{l,k}
        2. Boundary:    dE_rep/dR_A|_bnd = −z · R_A^2 · 4π · Σ_l w_l ρ_l n_{l,k}
        3. Volume:      z · d/dR Tr[P · S_out]
        """
        p_au = self.p_gpa / EH_BOHR3_TO_GPA  # pressure in atomic units
        
        # Surface integrals (cavitation + boundary)
        surface_data = _get_cavity_surface_grid(
            self.mol, radii_table=self.radii_table,
            lebedev_order=self.lebedev_order)
        
        grad = np.zeros((self.mol.natm, 3))
        for atom in range(self.mol.natm):
            sd = surface_data[atom]
            R  = sd['radius']
            w  = sd['weights']
            n  = sd['normals']
            prefactor = R**2 * 4.0 * np.pi
            
            # Cavitation: p · dV/dR_A
            wn = np.einsum('l,lk->k', w, n)
            grad[atom] += p_au * prefactor * wn
            
            # Boundary response of h_rep
            if len(sd['coords']) > 0:
                ao  = dft.numint.eval_ao(self.mol, sd['coords'], deriv=0)
                rho = np.einsum('lm,lm->l', ao @ dm, ao)
                grad[atom] -= self.z * prefactor * np.einsum('l,l,lk->k', w, rho, n)
        
        # Volume-integral contribution from h_rep
        grad += self.z * _get_S_out_nuc_grad(
            self.mol, dm, radii_table=self.radii_table,
            grid_level=self.grid_level, radial_grid_pts=self.radial_grid_pts)
        
        return grad

class _SCFWithXPCav(_SCFWithXP):
    """
    SCF wrapper that includes both repulsion and cavitation contributions,  
    along with analytical gradients for geometry optimisation under fixed pressure.
    """

    def get_e_cav(self, mol=None):
        """Get cavitation energy."""
        return self.with_xp.get_e_cav(mol)

    def grad(self, dm=None, mol=None):
        """Compute nuclear gradients."""
        if dm is None:
            dm = self.make_rdm1()
        return self.with_xp.grad(dm, mol)

    def energy_tot(self, dm=None, h1e=None, vhf=None):
        """Total energy including cavitation energy."""
        e_tot = super().energy_tot(dm, h1e, vhf)
        e_cav = self.with_xp.get_e_cav()
        e_tot += e_cav
        self.scf_summary['e_cav'] = e_cav
        return e_tot

    def nuc_grad_method(self):
        """Return gradient object supporting XPCav gradients."""
        # Unwrap to get underlying gradient object
        inner_grad = self.undo_xp().nuc_grad_method()
        # Store reference for XPCav gradient computation
        inner_grad.base = self
        # Wrap with XPCav gradient layer
        grad_obj = WithXPCavGrad(inner_grad)
        # Create dynamic class with appropriate name
        name = 'XPCav' + inner_grad.__class__.__name__
        return lib.set_class(grad_obj, (WithXPCavGrad, inner_grad.__class__), name)

    Gradients = nuc_grad_method

    def dump_flags(self, verbose=None):
        super().dump_flags(verbose)
        xp = self.with_xp
        log = lib.logger.new_logger(self, verbose)
        log.info('')
        log.info('******** XPCav parameters (geometry optimisation) ********')
        log.info('z (barrier height)    = %.3f', xp.z)
        log.info('p (pressure)          = %.4f GPa', xp.p_gpa)
        log.info('grid_level            = %d', xp.grid_level)
        log.info('lebedev_order         = %d', xp.lebedev_order)
        return self

class WithXPCavGrad:
    """
    Gradient mixin for XPCav — adds cavitation + boundary + volume
    gradient to whatever gradient object is underneath (vanilla SCF, PCM, …).
    """
    _keys = {'de_xp'}

    def __init__(self, grad_method):
        self.__dict__.update(grad_method.__dict__)
        self.de_xp = None

    def undo_xp(self):
        cls = self.__class__
        obj = lib.view(self, lib.drop_class(cls, WithXPCavGrad, 'XPCav'))
        del obj.de_xp
        return obj

    def kernel(self, *args, dm=None, **kwargs):
        if dm is None:
            dm = self.base.make_rdm1()
        dm_tot = dm[0] + dm[1] if np.shape(dm)[0] == 2 else dm

        # XPCav gradient: cavitation + boundary + volume
        self.de_xp = self.base.with_xp.grad(dm_tot)

        # Inner gradient (vanilla SCF, and/or PCM)
        de_inner = super().kernel(*args, **kwargs)

        self.de = de_inner + self.de_xp
        return self.de

    grad = kernel

def XPCav(mf, z, radii_table, p_gpa=None,
          grid_level=3, lebedev_order=29, radial_grid_pts=1000):
    """
    Attach XPCav (Pauli repulsion + cavitation) to an SCF method
    for geometry optimisation under fixed pressure.
    
    Usage:
        ks = dft.RKS(mol)
        ks_xpcav = XPCav(ks, z=0.5, radii_table=Rahm, p_gpa=0.1)
        ks_xpcav.kernel()  # SCF
        grad_solver = ks_xpcav.nuc_grad_method()  # Get gradient solver
        grad_solver.kernel()  # Compute gradients
    
    Parameters
    ----------
    mf : pyscf.scf.SCF
        SCF object to wrap.
    z : float
        Barrier height (Hartree).
    radii_table : array
        Pre-scaled cavity radii in Bohr, indexed by nuclear charge.
    p_gpa : float
        External pressure in GPa.
    grid_level : int, default 3
        PySCF grid level for volume and overlap integrals.
    lebedev_order : int, default 29
        Lebedev quadrature order for surface integrals (→ 302 points).
    radial_grid_pts : int or None, default 1000
        Number of radial grid points. If None, use grid_level default.
    
    Returns
    -------
    Modified SCF object with XPCav corrections (h_rep + e_cav) and gradients
    """
    if p_gpa is None:
        raise ValueError("p_gpa (pressure in GPa) is required for XPCav")
    
    if isinstance(mf, _SCFWithXP):
        raise ValueError(
            "Cannot attach XPCav to an object that already has XP.")
    
    with_xpcav = _XPCav(
        mf.mol, z, radii_table=radii_table,
        p_gpa=p_gpa, grid_level=grid_level, lebedev_order=lebedev_order,
        radial_grid_pts=radial_grid_pts)
    
    sol_mf = _SCFWithXPCav(mf, with_xpcav)
    name = 'XPCav' + mf.__class__.__name__
    return lib.set_class(sol_mf, (_SCFWithXPCav, mf.__class__), name)

# ---------------------------------------------------------------------------
#  Method-style convenience wrappers on SCF objects
# ---------------------------------------------------------------------------

def _xp_method(self, z, radii_table,
               grid_level=3, radial_grid_pts=1000):
    """Method-style XP wrapper: ks = ks.XP(...)."""
    return XP(
        self, z,
        radii_table=radii_table,
        grid_level=grid_level,
        radial_grid_pts=radial_grid_pts
    )

def _xpcav_method(self, z, radii_table, p_gpa=None,
                  grid_level=3, lebedev_order=29, radial_grid_pts=1000):
    """Method-style XPCav wrapper: ks = ks.XPCav(...)."""
    return XPCav(
        self, z,
        radii_table=radii_table,
        p_gpa=p_gpa,
        grid_level=grid_level,
        lebedev_order=lebedev_order,
        radial_grid_pts=radial_grid_pts
    )

def _patch_scf_factory_methods():
    """Enable ks.XP(...) and ks.XPCav(...) on PySCF SCF/KS objects."""
    from pyscf.scf import hf as scf_hf

    if getattr(scf_hf.SCF, "_xppcm_factory_methods_patched", False):
        return

    scf_hf.SCF.XP = _xp_method
    scf_hf.SCF.XPCav = _xpcav_method
    scf_hf.SCF._xppcm_factory_methods_patched = True

# ---------------------------------------------------------------------------
#  XP for post-HF (CCSD, CISD)
# ---------------------------------------------------------------------------

def _get_post_hf_dm_tot(method_obj):
    """Helper to compute the total AO density matrix from a post-HF method object."""
    mf = method_obj._scf
    unrestricted = any(m in mf.__class__.__name__ for m in ('UHF', 'UKS'))
    if unrestricted:
        dm_mo_a, dm_mo_b = method_obj.make_rdm1()
        dm_a = np.einsum('pi,ij,qj->pq', method_obj.mo_coeff[0], dm_mo_a, method_obj.mo_coeff[0].conj())
        dm_b = np.einsum('pi,ij,qj->pq', method_obj.mo_coeff[1], dm_mo_b, method_obj.mo_coeff[1].conj())
        return dm_a + dm_b
    else:
        dm_mo = method_obj.make_rdm1()
        return np.einsum('pi,ij,qj->pq', method_obj.mo_coeff, dm_mo, method_obj.mo_coeff.conj())

def _get_e_rep_post_hf(self, dm=None, mol=None):
        """Computes the Pauli repulsion energy for a post-HF method with XP attached."""
        if not hasattr(self, '_scf') or getattr(self._scf, 'with_xp', None) is None:
            raise AttributeError("Underlying SCF object does not use XP-PCM.")

        mf = self._scf
        xp_obj = mf.with_xp
        mol = mol or self.mol
        post_name = self.__class__.__name__

        if not any(m in post_name for m in ('CISD', 'CCSD')):
            raise ValueError(
                f"Post-HF method must contain one of ('CISD', 'CCSD'). Found: {post_name}")

        if dm is None: 
            dm = _get_post_hf_dm_tot(self)

        return xp_obj.get_e_rep(dm, mol=mol)

def _patch_post_scf_kernel_for_xp(cls):
    """Patch a post-SCF class to allow get_e_rep access."""
    # Allow direct access to .with_xp and .get_e_rep on post-HF objects
    if not hasattr(cls, 'with_xp'):
        setattr(cls, 'with_xp', property(lambda self: getattr(self._scf, 'with_xp', None)))
    if not hasattr(cls, 'get_e_rep'):
        setattr(cls, 'get_e_rep', _get_e_rep_post_hf)
 
def _patch_post_scf_kernels_for_xp():
    """Allow access to get_e_rep to supported post-SCF methods."""
    _patch_post_scf_kernel_for_xp(ci.cisd.CISD)
    _patch_post_scf_kernel_for_xp(ci.ucisd.UCISD)
    _patch_post_scf_kernel_for_xp(cc.ccsd.CCSD)
    _patch_post_scf_kernel_for_xp(cc.uccsd.UCCSD)
    _patch_post_scf_kernel_for_xp(cc.rccsd.RCCSD)
    # CCSDT or higher not supported because cannot get DM.

# ---------------------------------------------------------------------------
# Patch at import time
# ---------------------------------------------------------------------------

_patch_post_scf_kernels_for_xp()
_patch_scf_factory_methods()