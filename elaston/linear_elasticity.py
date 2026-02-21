# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np

from elaston import dislocation, elastic_constants, inclusion, tools

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


class LinearElasticity:
    """
    .. math:
        \\sigma_{ij} = C_{ijkl} * \\epsilon_{kl}

    where :math:\\sigma_{ij} is the ij-component of stress and
    :math:\\epsilon_{kl} is the kl-component of strain.

    Examples I: Get bulk modulus from the elastic tensor from the voigt average:

    >>> medium = LinearElasticity(C_11=211.0, C_12=130.0, C_44=82.0)
    >>> medium_voigt = medium.get_voigt_average()
    >>> parameters = medium_voigt.get_elastic_moduli()
    >>> print(parameters['bulk_modulus'])

    Example II: Get strain field around a point defect:

    >>> import numpy as np
    >>> medium = LinearElasticity(C_11=211.0, C_12=130.0, C_44=82.0)  # Fe
    >>> random_positions = np.random.random((10, 3)) - 0.5
    >>> dipole_tensor = np.eye(3)
    >>> print(medium.get_point_defect_strain(random_positions, dipole_tensor))

    Example III: Get stress field around a dislocation:

    >>> import numpy as np
    >>> medium = LinearElasticity(C_11=211.0, C_12=130.0, C_44=82.0)
    >>> random_positions = np.random.random((10, 3))-0.5
    >>> # Burgers vector of a screw dislocation in bcc Fe
    >>> burgers_vector = np.array([0, 0, 2.86 * np.sqrt(3) / 2])
    >>> print(medium.get_dislocation_stress(random_positions, burgers_vector))

    Example IV: Estimate the distance between partial dislocations:

    >>> medium = LinearElasticity(C_11=110.5, C_12=64.8, C_44=30.9)  # Al
    >>> lattice_constant = 4.05
    >>> partial_one = np.array([-0.5, 0, np.sqrt(3) / 2]) * lattice_constant
    >>> partial_two = np.array([0.5, 0, np.sqrt(3) / 2]) * lattice_constant
    >>> distance = 100
    >>> stress_one = medium.get_dislocation_stress([0, distance, 0], partial_one)
    >>> print('Choose `distance` in the way that the value below corresponds to SFE')
    >>> medium.get_dislocation_force(stress_one, [0, 1, 0], partial_two)
    """

    def __init__(
        self,
        C_tensor: np.ndarray | list | None = None,
        C_11: float | None = None,
        C_12: float | None = None,
        C_13: float | None = None,
        C_22: float | None = None,
        C_33: float | None = None,
        C_44: float | None = None,
        C_55: float | None = None,
        C_66: float | None = None,
        youngs_modulus: float | None = None,
        poissons_ratio: float | None = None,
        shear_modulus: float | None = None,
        orientation: np.ndarray | None = None,
    ) -> None:
        """
        Args:
            C_tensor ((6, 6)-array, (3, 3, 3, 3)-array): Elastic tensor in
                Voigt notation or full matrix
            C_11 (float): Elastic constant
            C_12 (float): Elastic constant
            C_13 (float): Elastic constant
            C_22 (float): Elastic constant
            C_33 (float): Elastic constant
            C_44 (float): Elastic constant
            C_55 (float): Elastic constant
            C_66 (float): Elastic constant
            youngs_modulus (float): Young's modulus
            poissons_ratio (float): Poisson's ratio
            shear_modulus (float): Shear modulus
            orientation ((3,3)-array): Rotation matrix that defines the orientation
                of the system. If set, the elastic tensor will be rotated accordingly.

        Here is a list of elastic constants of a few materials for
        :math:`C_{11}`, :math:`C_{12}`, and :math:`C_{44}` in GPa:

        - Al: 110.5, 64.8, 30.9
        - Cu: 170.0, 121.0, 75.0
        - Fe: 211.0, 130.0, 82.0
        - Mo: 262.0, 135.0, 120.0
        - Ni: 243.0, 160.0, 140.0
        - W: 411.0, 248.0, 160.0
        """
        self._orientation: np.ndarray | None = None
        self._elastic_tensor: np.ndarray = tools.C_from_voigt(
            elastic_constants.initialize_elastic_tensor(
                C_tensor=C_tensor,
                C_11=C_11,
                C_12=C_12,
                C_13=C_13,
                C_22=C_22,
                C_33=C_33,
                C_44=C_44,
                C_55=C_55,
                C_66=C_66,
                youngs_modulus=youngs_modulus,
                poissons_ratio=poissons_ratio,
                shear_modulus=shear_modulus,
            )
        )
        if orientation is not None:
            self.orientation = orientation

    @property
    def orientation(self) -> np.ndarray | None:
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
    def orientation(self, r: np.ndarray) -> None:
        self._orientation = tools.orthonormalize(r)

    def get_elastic_tensor(
        self, voigt: bool = False, rotate: bool = True
    ) -> np.ndarray:
        C: np.ndarray = self._elastic_tensor.copy()
        if self.orientation is not None and rotate:
            C = tools.crystal_to_box(C, self.orientation)
        if voigt:
            C = tools.C_to_voigt(C)
        return C

    def get_compliance_tensor(
        self, rotate: bool = True, voigt: bool = False
    ) -> np.ndarray:
        return tools.get_compliance_tensor(
            self.get_elastic_tensor(voigt=True, rotate=rotate), voigt=voigt
        )

    def get_zener_ratio(self) -> float:
        """
        Zener ratio or the anisotropy index. If 1, the medium is isotropic. If
        isotropic, the analytical form of the Green's function is used for the
        calculation of strain and displacement fields.
        """
        return elastic_constants.get_zener_ratio(
            self.get_elastic_tensor(voigt=True, rotate=False)
        )

    def is_cubic(self) -> bool:
        return elastic_constants.is_cubic(
            self.get_elastic_tensor(voigt=True, rotate=False)
        )

    def is_isotropic(self) -> bool:
        return elastic_constants.is_isotropic(
            self.get_elastic_tensor(voigt=True, rotate=False)
        )

    def get_elastic_moduli(self) -> dict[str, float]:
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
        if not self.is_isotropic():
            raise ValueError(
                "The material must be isotropic. Re-instantiate with isotropic"
                " elastic constants or run an averaging method"
                " (get_voigt_average, get_reuss_average) first"
            )
        return elastic_constants.get_elastic_moduli(
            self.get_elastic_tensor(voigt=True, rotate=False)
        )

    def get_point_defect_displacement(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
        check_unique: bool = False,
    ) -> np.ndarray:
        return inclusion.get_point_defect_displacement(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
            check_unique=check_unique,
        )

    get_point_defect_displacement.__doc__ += inclusion.point_defect_explanation

    def get_point_defect_strain(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
        check_unique: bool = False,
    ) -> np.ndarray:
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
        return inclusion.get_point_defect_strain(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
            check_unique=check_unique,
        )

    get_point_defect_strain.__doc__ += inclusion.point_defect_explanation

    def get_point_defect_stress(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
    ) -> np.ndarray:
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
        return inclusion.get_point_defect_stress(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
        )

    get_point_defect_stress.__doc__ += inclusion.point_defect_explanation

    def get_point_defect_energy_density(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
    ) -> np.ndarray:
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
        return inclusion.get_point_defect_energy_density(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
        )

    get_point_defect_energy_density.__doc__ += inclusion.point_defect_explanation

    def get_dislocation_displacement(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
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
        return dislocation.get_dislocation_displacement(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_strain(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
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
        return dislocation.get_dislocation_strain(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_stress(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
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
        return dislocation.get_dislocation_stress(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_energy_density(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
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
        return dislocation.get_dislocation_energy_density(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_energy(
        self,
        burgers_vector: np.ndarray,
        r_min: float,
        r_max: float,
        mesh: int = 100,
    ) -> float:
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
        return dislocation.get_dislocation_energy(
            self.get_elastic_tensor(), burgers_vector, r_min, r_max, mesh
        )

    @staticmethod
    def get_dislocation_force(
        stress: np.ndarray,
        glide_plane: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
        """
        Force per unit length along the dislocation line. This method is
        useful for estaimting the distance between partial dislocations. At
        equilibrium, the force acting on the dislocation line corresponds to
        the stacking fault energy (SFE).

        Args:
            stress ((n, 3, 3)-array): External stress field at the dislocation line
            glide_plane ((3,)-array): Glide plane
            burgers_vector ((3,)-array): Burgers vector

        Returns:
            ((3,)-array): Force per unit length acting on the dislocation.

        Here is a short list of the stacking fault energies of a few materials:

        - Ni: 90 mJ/m^2 = 5.62 meV/Å^2
        - Ag: 25 mJ/m^2 = 1.56 meV/Å^2
        - Au: 75 mJ/m^2 = 4.68 meV/Å^2
        - Cu: 70-78 mJ/m^2 = 4.36-4.87 meV/Å^2
        - Mg: 125 mJ/m^2 = 7.80 meV/Å^2
        - Al: 160-250 mJ/m^2 = 10.0-15.6 meV/Å^2

        Source: https://en.wikipedia.org/wiki/Stacking-fault_energy
        """
        return dislocation.get_dislocation_force(stress, glide_plane, burgers_vector)

    def get_voigt_average(self) -> "LinearElasticity":
        """
        Voigt average of the elastic tensor. The Voigt average is defined as
        the arithmetic mean of the elastic tensor. The Voigt average is
        isotropic and can be used to calculate the elastic moduli of the medium.
        """
        return LinearElasticity(
            **elastic_constants.get_voigt_average(self.get_elastic_tensor(voigt=True))
        )

    def get_reuss_average(self) -> "LinearElasticity":
        """
        Reuss average of the elastic tensor. The Reuss average is obtained from
        the arithmetic mean of the compliance tensor. The Reuss average is
        isotropic and can be used to calculate the elastic moduli of the medium.
        """
        return LinearElasticity(
            **elastic_constants.get_reuss_average(self.get_elastic_tensor(voigt=True))
        )
