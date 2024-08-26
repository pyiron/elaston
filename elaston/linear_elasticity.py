# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from typing import Optional
from elaston.green import Anisotropic, Isotropic, Green
from elaston.eshelby import Eshelby
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
>>> dipole_tensor = -job.structure.get_volume()*job['output/generic/pressures'][-1]

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


class LinearElasticity:
    """
    Linear elastic field class based on the 3x3x3x3 elastic tensor :math:C_{ijkl}:

    .. math:
        \\sigma_{ij} = C_{ijkl} * \\epsilon_{kl}

    where :math:\\sigma_{ij} is the ij-component of stress and
    :math:\\epsilon_{kl} is the kl-component of strain.

    Examples I: Get bulk modulus from the elastic tensor:

    >>> medium = LinearElasticity(elastic_tensor)
    >>> print(medium.bulk_modulus)

    Example II: Get strain field around a point defect:

    >>> import numpy as np
    >>> medium = LinearElasticity(elastic_tensor)
    >>> random_positions = np.random.random((10, 3))-0.5
    >>> dipole_tensor = np.eye(3)
    >>> print(medium.get_point_defect_strain(random_positions, dipole_tensor))

    Example III: Get stress field around a dislocation:

    >>> import numpy as np
    >>> medium = LinearElasticity(elastic_tensor)
    >>> random_positions = np.random.random((10, 3))-0.5
    >>> burgers_vector = np.array([0, 0, 1])
    >>> print(medium.get_dislocation_stress(random_positions, burgers_vector))

    Example IV: Estimate the distance between partial dislocations:

    >>> medium = LinearElasticity(elastic_tensor)
    >>> partial_one = np.array([-0.5, 0, np.sqrt(3)/2])*lattice_constant
    >>> partial_two = np.array([0.5, 0, np.sqrt(3)/2])*lattice_constant
    >>> distance = 100
    >>> stress_one = medium.get_dislocation_stress([0, distance, 0], partial_one)
    >>> print('Choose `distance` in the way that the value below corresponds to SFE')
    >>> medium.get_dislocation_force(stress_one, [0, 1, 0], partial_two)

    """

    def __init__(
        self,
        C_tensor=None,
        C_11=None,
        C_12=None,
        C_13=None,
        C_22=None,
        C_33=None,
        C_44=None,
        C_55=None,
        C_66=None,
        youngs_modulus=None,
        poissons_ratio=None,
        shear_modulus=None,
        orientation=None,
    ):
        """
        Args:
            elastic_tensor ((3,3,3,3)-, (6,6)- or (3,)-array): Elastic tensor
                (in C_ijkl notation, Voigt notation or a 3-component array
                containing [C_11, C_12, C_44]).
            orientation ((3,3)-array): Rotation matrix that defines the
                orientation of the system.

        Here is a list of elastic constants of a few materials:

        - Al: [110.5, 64.8, 30.9]
        - Cu: [170.0, 121.0, 75.0]
        - Fe: [211.0, 130.0, 82.0]
        - Mo: [442.0, 142.0, 162.0]
        - Ni: [248.0, 140.0, 76.0]
        - W: [630.0, 161.0, 160.0]
        """
        self._orientation = None
        self._elastic_tensor = tools.C_from_voigt(
            elastic_constants.initialize_elastic_tensor(elastic_tensor)
        )
        if orientation is not None:
            self.orientation = orientation

    @property
    def orientation(self):
        """
        Rotation matrix that defines the orientation of the system. If set,
        the elastic tensor will be rotated accordingly. For example a box with
        a dislocation should get:

        >>> medium.orientation = np.array([[1, 1, 1], [1, 0, -1], [1, -2, 1]])

        If a non-orthogonal orientation is set, the second vector is
        orthogonalized with the Gram Schmidt process. It is not necessary to
        specify the third axis as it is automatically calculated.
        """
        return self._orientation

    @orientation.setter
    def orientation(self, r):
        self._orientation = tools.orthonormalize(r)

    def get_elastic_tensor(self, voigt=False, orientation=None):
        C = self._elastic_tensor.copy()
        if orientation is not None:
            orientation = tools.orthonormalize(orientation)
        elif self.orientation is not None:
            orientation = self.orientation
        if orientation is not None:
            C = tools.crystal_to_box(C, orientation)
        if voigt:
            C = tools.C_to_voigt(C)
        return C

    def get_compliance_matrix(self, orientation=None):
        """Compliance matrix in Voigt notation."""
        return np.linalg.inv(
            self.get_elastic_tensor(voigt=True, orientation=orientation)
        )

    def get_zener_ratio(self):
        """
        Zener ratio or the anisotropy index. If 1, the medium is isotropic. If
        isotropic, the analytical form of the Green's function is used for the
        calculation of strain and displacement fields.
        """
        return elastic_constants.get_zener_ratio(
            tools.C_to_voigt(self._elastic_tensor)
        )

    def is_isotropic(self):
        return elastic_constants.is_isotropic(self.get_elastic_tensor(voigt=True))

    def get_elastic_moduli(self):
        if not self.is_isotropic():
            raise ValueError(
                "The material must be isotropic. Re-instantiate with isotropic"
                " elastic constants or run an averaging method"
                " (get_voigt_average, get_reuss_average) first"
            )
        return elastic_tensor.get_elastic_moduli(
            tools.C_to_voigt(self._elastic_tensor)
        )

    def get_greens_function(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
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
            isotropic (bool): Whether to use the isotropic or anisotropic
                elasticity. If the medium is isotropic, it will automatically
                be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`
            check_unique (bool): Whether to check the unique positions

        Returns:
            ((n,3,3)-array): Green's function values for the given positions
        """
        if isotropic and self.is_isotropic():
            param = self.get_elastic_moduli()
            C = Isotropic(
                param["poissons_ratio"], param["shear_modulus"], optimize=optimize
            )
        else:
            C = Anisotropic(
                self.get_elastic_tensor(orientation=orientation),
                n_mesh=n_mesh,
                optimize=optimize
            )
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
        orientation: Optional[np.ndarray] = None,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        isotropic: bool = False,
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
            isotropic (bool): Whether to use the isotropic or anisotropic
                elasticity. If the medium is isotropic, it will automatically
                be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`
            check_unique (bool): Whether to check the unique positions

        Returns:
            ((n,3)-array): Displacement field
        """
        g_tmp = self.get_greens_function(
            positions,
            orientation=orientation,
            derivative=1,
            fourier=False,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize,
            check_unique=check_unique,
        )
        return -np.einsum("...ijk,...jk->...i", g_tmp, dipole_tensor)

    get_point_defect_displacement.__doc__ += point_defect_explanation

    def get_point_defect_strain(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        isotropic: bool = False,
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
            isotropic (bool): Whether to use the isotropic or anisotropic
                elasticity. If the medium is isotropic, it will automatically
                be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`
            check_unique (bool): Whether to check the unique positions

        Returns:
            ((n,3,3)-array): Strain field
        """
        g_tmp = self.get_greens_function(
            positions,
            orientation=orientation,
            derivative=2,
            fourier=False,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize,
            check_unique=check_unique,
        )
        v = -np.einsum("...ijkl,...kl->...ij", g_tmp, dipole_tensor)
        return 0.5 * (v + np.einsum("...ij->...ji", v))

    get_point_defect_strain.__doc__ += point_defect_explanation

    def get_point_defect_stress(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        isotropic: bool = False,
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
            isotropic (bool): Whether to use the isotropic or anisotropic
                elasticity. If the medium is isotropic, it will automatically
                be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,3,3)-array): Stress field
        """
        strain = self.get_point_defect_strain(
            positions=positions,
            orientation=orientation,
            dipole_tensor=dipole_tensor,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize,
        )
        return np.einsum(
            "ijkl,...kl->...ij",
            self.get_elastic_tensor(orientation=orientation),
            strain
        )

    get_point_defect_stress.__doc__ += point_defect_explanation

    def get_point_defect_energy_density(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        isotropic: bool = False,
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
            isotropic (bool): Whether to use the isotropic or anisotropic
                elasticity. If the medium is isotropic, it will automatically
                be set to isotropic=True
            optimize (bool): cf. `optimize` in `numpy.einsum`

        Returns:
            ((n,)-array): Energy density field
        """
        strain = self.get_point_defect_strain(
            positions=positions,
            orientation=orientation,
            dipole_tensor=dipole_tensor,
            n_mesh=n_mesh,
            isotropic=isotropic,
            optimize=optimize,
        )
        return np.einsum(
            "ijkl,...kl,...ij->...",
            self.get_elastic_tensor(orientation=orientation),
            strain,
            strain
        )

    get_point_defect_energy_density.__doc__ += point_defect_explanation

    def get_dislocation_displacement(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        burgers_vector: np.ndarray,
    ):
        """
        Displacement field around a dislocation according to anisotropic
        elasticity theory described by
        [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation.
                The third axis coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n, 3)-array): Displacement field (z-axis coincides with the
                dislocation line)
        """
        eshelby = Eshelby(
            self.get_elastic_tensor(orientation=orientation), burgers_vector
        )
        return eshelby.get_displacement(positions)

    def get_dislocation_strain(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        burgers_vector: np.ndarray,
    ):
        """
        Strain field around a dislocation according to anisotropic elasticity theory
        described by [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation.
                The third axis coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n, 3, 3)-array): Strain field (z-axis coincides with the dislocation line)
        """
        eshelby = Eshelby(
            self.get_elastic_tensor(orientation=orientation), burgers_vector
        )
        return eshelby.get_strain(positions)

    def get_dislocation_stress(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        burgers_vector: np.ndarray,
    ):
        """
        Stress field around a dislocation according to anisotropic elasticity theory
        described by [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation.
                The third axis coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n, 3, 3)-array): Stress field (z-axis coincides with the dislocation line)
        """
        strain = self.get_dislocation_strain(
            positions, orientation=orientation, burgers_vector=burgers_vector
        )
        return np.einsum(
            "ijkl,...kl->...ij", self.get_elastic_tensor(orientation), strain
        )

    def get_dislocation_energy_density(
        self,
        positions: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        burgers_vector: np.ndarray,
    ):
        """
        Energy density field around a dislocation (product of stress and
        strain, cf. corresponding methods)

        Args:
            positions ((n,2) or (n,3)-array): Position around a dislocation.
                The third axis coincides with the dislocation line.
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((n,)-array): Energy density field
        """
        strain = self.get_dislocation_strain(positions, orientation, burgers_vector)
        return np.einsum(
            "ijkl,...kl,...ij->...",
            self.get_elastic_tensor(orientation),
            strain,
            strain
        )

    def get_dislocation_energy(
        self,
        burgers_vector: np.ndarray,
        orientation: Optional[np.ndarray] = None,
        r_min: float,
        r_max: float,
        mesh: int = 100,
    ):
        """
        Energy per unit length along the dislocation line.

        Args:
            burgers_vector ((3,)-array): Burgers vector
            r_min (float): Minimum distance from the dislocation core
            r_max (float): Maximum distance from the dislocation core
            mesh (int): Number of grid points for the numerical integration
                along the angle

        Returns:
            (float): Energy of dislocation per unit length

        The energy is defined by the product of the stress and strain (i.e.
        energy density), which is integrated over the plane vertical to the
        dislocation line. The energy density :math:`w` according to the linear
        elasticity is given by:

        .. math:
            w(r, \\theta) = A(\\theta)/r^2

        Therefore, the energy per unit length :math:`U` is given by:

        .. math:
            U = \\log(r_max/r_min)\\int A(\\theta)\\mathrm d\\theta

        This implies :math:`r_min` cannot be 0 as well as :math:`r_max` cannot
        be infinity. This is the consequence of the fact that the linear
        elasticity cannot describe the core structure properly, and a real
        medium is not infinitely large. While :math:`r_max` can be defined
        based on the real dislocation density, the choice of :math:`r_min`
        should be done carefully.
        """
        if r_min <= 0:
            raise ValueError("r_min must be a positive float")
        theta_range = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        r = np.stack((np.cos(theta_range), np.sin(theta_range)), axis=-1) * r_min
        strain = self.get_dislocation_strain(r, burgers_vector=burgers_vector)
        return np.einsum(
            "ijkl,nkl,nij->", self.get_elastic_tensor(orientation), strain, strain
        ) / np.diff(theta_range)[0] * r_min**2 * np.log(r_max / r_min)

    @staticmethod
    def get_dislocation_force(
        stress: np.ndarray,
        glide_plane: np.ndarray,
        burgers_vector: np.ndarray,
    ):
        """
        Force per unit length along the dislocation line.

        Args:
            stress ((n, 3, 3)-array): External stress field at the dislocation line
            glide_plane ((3,)-array): Glide plane
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((3,)-array): Force per unit length acting on the dislocation.
        """
        g = np.asarray(glide_plane) / np.linalg.norm(glide_plane)
        return np.einsum(
            "i,...ij,j,k->...k", g, stress, burgers_vector, np.cross(g, [0, 0, 1])
        )
