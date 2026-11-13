#!/usr/bin/env python
#
# Author: Erika Magnusson <ermag@chalmers.se>
#

'''
Basis-set augmentation for the XP-PCM cavity wall.


Outside the cavity the orbitals meet the barrier Z and decay as exp(-kappa r)
with kappa = sqrt(2 Z). Standard basis sets are built for free molecules and
cannot bend there, and the error grows with the barrier height. augment_basis()
adds even-tempered s and p shells centred on the exponent that matches this
decay at the cavity sphere of each element, a* = kappa / (2 R). This effect
depends on the barrier height Z, so for higher compressions the effect is
noticable.

This example compares the XP energy and the analytical pressure of water from
cc-pVDZ and cc-pVTZ, with and without the added shells, at three compressions.
Without them the pressures of the two basis sets differ by 10-40 %; with them
the two agree to within a few percent.

a* depends on Z and on the cavity radii, so the molecule is augmented again at
every point of a compression scan, with the same Z, radii and grid_level that
are passed to XP.
'''

from pyscf import gto, dft
from pyscf import xppcm   # registers mf.XP() and mf.XPCav() on SCF objects
from pyscf.xppcm import (Rahm, get_z0, get_z, get_cavity_volume,
                         calculate_analytical_pressure, augment_basis)

# Medium: cyclohexane
z0 = get_z0(36, 0.7781, 84.1595)   # valence electrons, density (g/cm^3), molar mass (g/mol)
f_0 = 1.30                         # ambient cavity scale, defines V_0

geometry = 'O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469'


def xp_point(mol, z, radii, volume):
    '''XP energy (Ha) and analytical pressure (GPa).'''
    mf = dft.RKS(mol, xc='pbe0').XP(z=z, radii_table=radii).run()
    return mf.e_tot, calculate_analytical_pressure(mf, volume)


# 1. The shells added at one point. They are reported at verbose >= 3; an
#    exponent the base set already covers is skipped, as is one too diffuse
#    for the exterior grid of XP.
mol = gto.M(atom=geometry, basis='cc-pvdz', verbose=0)
volume_0 = get_cavity_volume(mol, Rahm * f_0)
radii = Rahm * 0.9
z = get_z(get_cavity_volume(mol, radii), volume_0, z0)
mol_aug = augment_basis(mol, z, radii, verbose=3)

# 2. Energy and pressure with and without the added shells
print()
print('   f    z (Ha)  basis      nao    E (Ha)        p (GPa)   nao(aug)  E(aug) (Ha)   p(aug) (GPa)')
for f in (1.0, 0.9, 0.8):
    radii = Rahm * f
    volume = get_cavity_volume(mol, radii)
    z = get_z(volume, volume_0, z0)
    for basis in ('cc-pvdz', 'cc-pvtz'):
        mol = gto.M(atom=geometry, basis=basis, verbose=0)
        mol_aug = augment_basis(mol, z, radii)
        e, p = xp_point(mol, z, radii, volume)
        e_aug, p_aug = xp_point(mol_aug, z, radii, volume)
        print('  %.1f   %.3f   %-8s  %4d  %12.6f  %8.2f     %4d    %12.6f  %8.2f'
              % (f, z, basis, mol.nao, e, p, mol_aug.nao, e_aug, p_aug))

# 3. Higher angular momenta and a wider or denser window are available through
#    l_max, window and n; l_max=0 is enough for s-block atoms.
mol_aug = augment_basis(mol, z, radii, n=8, window=6.0, l_max=2, verbose=3)
