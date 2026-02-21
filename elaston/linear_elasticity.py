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
    Linear elastic field class based on the 3x3x3x3 elastic tensor :math:C_{ijkl}:
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

    def get_point_defect_strain(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
        check_unique: bool = False,
    ) -> np.ndarray:
        return inclusion.get_point_defect_strain(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
            check_unique=check_unique,
        )

    def get_point_defect_stress(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
    ) -> np.ndarray:
        return inclusion.get_point_defect_stress(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
        )

    def get_point_defect_energy_density(
        self,
        positions: np.ndarray,
        dipole_tensor: np.ndarray,
        n_mesh: int = 100,
        optimize: bool = True,
    ) -> np.ndarray:
        return inclusion.get_point_defect_energy_density(
            C=self.get_elastic_tensor(),
            x=positions,
            P=dipole_tensor,
            n_mesh=n_mesh,
            optimize=optimize,
        )

    def get_dislocation_displacement(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
        return dislocation.get_dislocation_displacement(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_strain(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
        return dislocation.get_dislocation_strain(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_stress(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
        return dislocation.get_dislocation_stress(
            self.get_elastic_tensor(), positions, burgers_vector
        )

    def get_dislocation_energy_density(
        self,
        positions: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
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
        return dislocation.get_dislocation_energy(
            self.get_elastic_tensor(), burgers_vector, r_min, r_max, mesh
        )

    @staticmethod
    def get_dislocation_force(
        stress: np.ndarray,
        glide_plane: np.ndarray,
        burgers_vector: np.ndarray,
    ) -> np.ndarray:
        return dislocation.get_dislocation_force(stress, glide_plane, burgers_vector)

    def get_voigt_average(self) -> "LinearElasticity":
        return LinearElasticity(
            **elastic_constants.get_voigt_average(self.get_elastic_tensor(voigt=True))
        )

    def get_reuss_average(self) -> "LinearElasticity":
        return LinearElasticity(
            **elastic_constants.get_reuss_average(self.get_elastic_tensor(voigt=True))
        )
