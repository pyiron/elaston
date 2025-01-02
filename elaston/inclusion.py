# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from elaston.green import get_greens_function
from semantikon.typing import u
from semantikon.converter import units as semantikon_units

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

where :math:`u_i(r)` is the displacement field of component :math:`i` at
position :math:`r` and :math:`f_j(a)` is the force component :math:`j` of the
atom at position :math:`a`. By taking the polynomial development we obtain:

.. math:
    u_i(r) \\approx G_{ij}(r)\\sum_a f_j(a)-\\frac{\\partial G_{ij}}{\\partial r_k}(r)\\sum_a a_k f_j(a)

The first term disappears because the sum of the forces is zero. From the
second term we define the dipole tensor :math:`P_{jk} = a_k f_j(a)`. Following
the definition above, we can obtain the displacement field, strain field,
stress field and energy density field if the dipole tensor and the elastic
tensor are known.

The dipole tensor of a point defect is commonly obtained from the following equation:

.. math:
    U = \\frac{V}{2} \\varepsilon_{ij}C_{ijkl}\\varepsilon_{kl}-P_{kl}\\varepsilon_{kl}

where :math:`U` is the potential energy, :math:`V` is the volume and
:math:`\\varepsilon` is the strain field. At equilibrium, the derivative of
the potential energy with respect to the strain disappears:

.. math:
    P_{ij} = VC_{ijkl}\\varepsilon_{kl} = V\\sigma_{ij}

With this in mind, we can calculate the dipole tensor of Ni in Al with the
following lines:

>>> from pyiron_atomistics import Project
>>> pr = Project('dipole_tensor')
>>> job = pr.create.job.Lammps('dipole')
>>> n_repeat = 3
>>> job.structure = pr.create.structure.bulk('Al', cubic=True).repeat(n_repeat)
>>> job.structure[0] = 'Ni'
>>> job.calc_minimize()
>>> job.run()
>>> dipole_tensor = -job.structure.get_volume() * job['output/generic/pressures'][-1]

Instead of working with atomistic calculations, the dipole tensor can be
calculated by the lambda tensor [1], which is defined as:

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


@semantikon_units
def get_point_defect_displacement(
    C: u(np.ndarray, units="=C"),
    x: u(np.ndarray, units="=x"),
    P: u(np.ndarray, units="=P"),
    n_mesh: int = 100,
    optimize: bool = True,
    check_unique: bool = False,
) -> u(np.ndarray, units="=P/C/x**2"):
    """
    Displacement field around a point defect

    Args:
        C ((3,3,3,3)-array): Elastic tensor
        x ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        P ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`
        check_unique (bool): Whether to check the unique positions

    Returns:
        ((n,3)-array): Displacement field
    """
    g_tmp = get_greens_function(
        C=C,
        x=x,
        derivative=1,
        fourier=False,
        n_mesh=n_mesh,
        optimize=optimize,
        check_unique=check_unique,
    )
    return -np.einsum("...ijk,...jk->...i", g_tmp, P)


@semantikon_units
def get_point_defect_strain(
    C: u(np.ndarray, units="=C"),
    x: u(np.ndarray, units="=x"),
    P: u(np.ndarray, units="=P"),
    n_mesh: int = 100,
    optimize: bool = True,
    check_unique: bool = False,
) -> u(np.ndarray, units="=P/C/x**3"):
    """
    Strain field around a point defect using the Green's function method

    Args:
        C ((3,3,3,3)-array): Elastic tensor
        x ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        P ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`
        check_unique (bool): Whether to check the unique positions

    Returns:
        ((n,3,3)-array): Strain field
    """
    g_tmp = get_greens_function(
        C=C,
        x=x,
        derivative=2,
        fourier=False,
        n_mesh=n_mesh,
        optimize=optimize,
        check_unique=check_unique,
    )
    v = -np.einsum("...ijkl,...kl->...ij", g_tmp, P)
    return 0.5 * (v + np.einsum("...ij->...ji", v))


@semantikon_units
def get_point_defect_stress(
    C: u(np.ndarray, units="=C"),
    x: u(np.ndarray, units="=x"),
    P: u(np.ndarray, units="=P"),
    n_mesh: int = 100,
    optimize: bool = True,
) -> u(np.ndarray, units="=P/C/x**3"):
    """
    Stress field around a point defect using the Green's function method

    Args:
        C ((3,3,3,3)-array): Elastic tensor
        x ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        P ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`

    Returns:
        ((n,3,3)-array): Stress field
    """
    strain = get_point_defect_strain(
        C=C,
        x=x,
        P=P,
        n_mesh=n_mesh,
        optimize=optimize,
    )
    return np.einsum("ijkl,...kl->...ij", C, strain)


@semantikon_units
def get_point_defect_energy_density(
    C: u(np.ndarray, units="=C"),
    x: u(np.ndarray, units="=x"),
    P: u(np.ndarray, units="=P"),
    n_mesh: int = 100,
    optimize: bool = True,
) -> u(np.ndarray, units="=P**2/C/x**6"):
    """
    Energy density field around a point defect using the Green's function method

    Args:
        C ((3,3,3,3)-array): Elastic tensor
        x ((n,3)-array): Positions in real space or reciprocal
            space (if fourier=True).
        P ((3,3)-array): Dipole tensor
        n_mesh (int): Number of mesh points in the radial integration in
            case if anisotropic Green's function (ignored if isotropic=True
            or fourier=True)
        optimize (bool): cf. `optimize` in `numpy.einsum`

    Returns:
        ((n,)-array): Energy density field
    """
    strain = get_point_defect_strain(
        C=C,
        x=x,
        P=P,
        n_mesh=n_mesh,
        optimize=optimize,
    )
    return np.einsum("ijkl,...kl,...ij->...", C, strain, strain)
