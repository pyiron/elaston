# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from elaston.green import Anisotropic, Isotropic, Green
from elaston import dislocation
from elaston import tools
from elaston import elastic_constants

__author__ = "Sam Waseda"
__copyright__ = (
    "Copyright 2021, Max-Planck-Institut für Eisenforschung GmbH "
    "- Computational Materials Design (CM) Department"
)
__version__ = "1.0"
__maintainer__ = "Sam Waseda"
__email__ = "waseda@mpie.de"
__status__ = "development"
__date__ = "Aug 21, 2021"


point_defect_explanation = """
According to the definition of the Green's function (cf. docstring of `get_greens_function`):

.. math:
    u_i(r) = \\sum_a G_{ij}(r-a)f_j(a)

where :math:`u_i(r)` is the displacement field of component :math:`i` at position :math:`r` and
:math:`f_j(a)` is the force component :math:`j` of the atom at position :math:`a`. By taking the
polynomial development we obtain:

.. math:
    u_i(r) \\approx G_{ij}(r)\\sum_a f_j(a)-\\frac{\\partial G_{ij}}{\\partial r_k}(r)\\sum_a a_k f_j(a)

The first term disappears because the sum of the forces is zero. From the second term we define
the dipole tensor :math:`P_{jk} = a_k f_j(a)`. Following the definition above, we can obtain the
displacement field, strain field, stress field and energy density field if the dipole tensor and
the elastic tensor are known.

The dipole tensor of a point defect is commonly obtained from the following equation:

.. math:
    U = \\frac{V}{2} \\varepsilon_{ij}C_{ijkl}\\varepsilon_{kl}-P_{kl}\\varepsilon_{kl}

where :math:`U` is the potential energy, :math:`V` is the volume and :math:`\\varepsilon` is the
strain field. At equilibrium, the derivative of the potential energy with respect to the strain
disappears:

.. math:
    P_{ij} = VC_{ijkl}\\varepsilon_{kl} = V\\sigma_{ij}

With this in mind, we can calculate the dipole tensor of Ni in Al with the following lines:

>>> from pyiron_atomistics import Project
>>> pr = Project('dipole_tensor')
>>> job = pr.create.job.Lammps('dipole')
>>> n_repeat = 3
>>> job.structure = pr.create.structure.bulk('Al', cubic=True).repeat(n_repeat)
>>> job.structure[0] = 'Ni'
>>> job.calc_minimize()
>>> job.run()
>>> dipole_tensor = -job.structure.get_volume() * job['output/generic/pressures'][-1]

Instead of working with atomistic calculations, the dipole tensor can be calculated by the
lambda tensor [1], which is defined as:

.. math:
    \\lambda_{ij} = \\frac{1]{V} \\frac{\\partial \\varepsilon_{ij}}{\\partial c}

where :math:`c` is the concentration of the defect, :math:`V` is the volume
and :math:`\\varepsilon` is the strain field. Then the dipole tensor is given by:

.. math:
    P_{ij} = VC_{ijkl}\\lambda_{kl}

ref:

[1]
Nowick, Arthur S.
Anelastic relaxation in crystalline solids.
Vol. 1. Elsevier, 2012.
"""


def get_greens_function(
    self,
    positions: np.ndarray,
    derivative: int = 0,
    fourier: bool = False,
    n_mesh: int = 100,
    optimize: bool = True,
    check_unique: bool = False,
):
    """
    Green's function of the equilibrium condition:

    C_ijkl d^2u_k/dx_jdx_l = 0

    Args:
        positions ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        derivative (int): 0th, 1st or 2nd derivative of the Green's
            function. Ignored if `fourier=True`.
        fourier (bool): If `True`,  the Green's function of the reciprocal
            space is returned.
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`
        check_unique (bool): Whether to check the unique positions

    Returns:
        ((n,3,3)-array): Green's function values for the given positions
    """
    if self.is_isotropic():
        param = self.get_elastic_moduli()
        C = Isotropic(
            param["poissons_ratio"], param["shear_modulus"], optimize=optimize
        )
    else:
        C = Anisotropic(self.get_elastic_tensor(), n_mesh=n_mesh, optimize=optimize)
    return C.get_greens_function(
        r=positions,
        derivative=derivative,
        fourier=fourier,
        check_unique=check_unique,
    )

get_greens_function.__doc__ += Green.__doc__

def get_point_defect_displacement(
    self,
    positions: np.ndarray,
    dipole_tensor: np.ndarray,
    n_mesh: int = 100,
    optimize: bool = True,
    check_unique: bool = False,
):
    """
    Displacement field around a point defect

    Args:
        positions ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        dipole_tensor ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`
        check_unique (bool): Whether to check the unique positions

    Returns:
        ((n,3)-array): Displacement field
    """
    g_tmp = self.get_greens_function(
        positions,
        derivative=1,
        fourier=False,
        n_mesh=n_mesh,
        optimize=optimize,
        check_unique=check_unique,
    )
    return -np.einsum("...ijk,...jk->...i", g_tmp, dipole_tensor)

get_point_defect_displacement.__doc__ += point_defect_explanation

def get_point_defect_strain(
    self,
    positions: np.ndarray,
    dipole_tensor: np.ndarray,
    n_mesh: int = 100,
    optimize: bool = True,
    check_unique: bool = False,
):
    """
    Strain field around a point defect using the Green's function method

    Args:
        positions ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        dipole_tensor ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`
        check_unique (bool): Whether to check the unique positions

    Returns:
        ((n,3,3)-array): Strain field
    """
    g_tmp = self.get_greens_function(
        positions,
        derivative=2,
        fourier=False,
        n_mesh=n_mesh,
        optimize=optimize,
        check_unique=check_unique,
    )
    v = -np.einsum("...ijkl,...kl->...ij", g_tmp, dipole_tensor)
    return 0.5 * (v + np.einsum("...ij->...ji", v))

get_point_defect_strain.__doc__ += point_defect_explanation

def get_point_defect_stress(
    self,
    positions: np.ndarray,
    dipole_tensor: np.ndarray,
    n_mesh: int = 100,
    optimize: bool = True,
):
    """
    Stress field around a point defect using the Green's function method

    Args:
        positions ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        dipole_tensor ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`

    Returns:
        ((n,3,3)-array): Stress field
    """
    strain = self.get_point_defect_strain(
        positions=positions,
        dipole_tensor=dipole_tensor,
        n_mesh=n_mesh,
        optimize=optimize,
    )
    return np.einsum("ijkl,...kl->...ij", self.get_elastic_tensor(), strain)

get_point_defect_stress.__doc__ += point_defect_explanation

def get_point_defect_energy_density(
    self,
    positions: np.ndarray,
    dipole_tensor: np.ndarray,
    n_mesh: int = 100,
    optimize: bool = True,
):
    """
    Energy density field around a point defect using the Green's function method

    Args:
        positions ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        dipole_tensor ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`

    Returns:
        ((n,)-array): Energy density field
    """
    strain = self.get_point_defect_strain(
        positions=positions,
        dipole_tensor=dipole_tensor,
        n_mesh=n_mesh,
        optimize=optimize,
    )
    return np.einsum(
        "ijkl,...kl,...ij->...", self.get_elastic_tensor(), strain, strain
    )

get_point_defect_energy_density.__doc__ += point_defect_explanation

