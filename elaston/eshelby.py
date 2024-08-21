# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from functools import cached_property

from elaston.units import units

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


class Eshelby:
    """
    Anisotropic elasticity theory for dislocations described by
    [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

    All notations follow the original paper.
    """

    def __init__(self, elastic_tensor, burgers_vector):
        """
        Args:
            elastic_tensor ((3,3,3,3)-array): Elastic tensor
            burgers_vector ((3,)-array): Burgers vector
        """
        assert np.shape(elastic_tensor) == (3, 3, 3, 3)
        assert np.shape(burgers_vector) == (3,)
        self.elastic_tensor = elastic_tensor
        self.burgers_vector = burgers_vector
        self.fit_range = np.linspace(0, 1, 10)

    def _get_pmat(self, x):
        return (
            self.elastic_tensor[:, 0, :, 0]
            + np.einsum(
                "...,ij->...ij",
                x,
                self.elastic_tensor[:, 0, :, 1] + self.elastic_tensor[:, 1, :, 0],
            )
            + np.einsum("...,ij->...ij", x**2, self.elastic_tensor[:, 1, :, 1])
        )

    @cached_property
    def p(self):
        coeff = np.polyfit(
            self.fit_range, np.linalg.det(self._get_pmat(self.fit_range)), 6
        )
        p = np.roots(coeff)
        p = p[np.imag(p) > 0]
        return p

    @cached_property
    def Ak(self):
        Ak = []
        for mat in self._get_pmat(self.p):
            values, vectors = np.linalg.eig(mat.T)
            Ak.append(vectors.T[np.absolute(values).argmin()])
        return np.array(Ak)

    @cached_property
    def D(self):
        F = np.einsum("n,ij->nij", self.p, self.elastic_tensor[:, 1, :, 1])
        F += self.elastic_tensor[:, 1, :, 0]
        F = np.einsum("nik,nk->ni", F, self.Ak)
        F = np.concatenate((F.T, self.Ak.T), axis=0)
        F = np.concatenate((np.real(F), -np.imag(F)), axis=-1)
        D = np.linalg.solve(F, np.concatenate((np.zeros(3), self.burgers_vector)))
        return D[:3] + 1j * D[3:]

    @property
    def dzdx(self):
        return np.stack((np.ones_like(self.p), self.p, np.zeros_like(self.p)), axis=-1)

    def _get_z(self, positions):
        z = np.stack((np.ones_like(self.p), self.p), axis=-1)
        return np.einsum("nk,...k->...n", z, np.asarray(positions)[..., :2])

    def get_displacement(self, positions):
        """
        Displacement vectors

        Args:
            positions ((n,3)-array): Positions for which the displacements are
                to be calculated

        Returns:
            ((n,3)-array): Displacement vectors
        """
        return np.imag(
            np.einsum(
                "nk,n,...n->...k", self.Ak, self.D, np.log(self._get_z(positions))
            )
        ) / (2 * np.pi)

    def get_strain(self, positions):
        """
        Strain tensors

        Args:
            positions ((n,3)-array): Positions for which the strains are to be
                calculated

        Returns:
            ((n,3,3)-array): Strain tensors
        """
        strain = np.imag(
            np.einsum(
                "ni,n,...n,nj->...ij",
                self.Ak,
                self.D,
                1 / self._get_z(positions),
                self.dzdx,
            )
        )
        strain = strain + np.einsum("...ij->...ji", strain)
        return strain / 4 / np.pi


@units(
    outputs=lambda burgers_vector: burgers_vector.u
)
def get_dislocation_displacement(
    elastic_tensor: np.ndarray,
    positions: np.ndarray,
    burgers_vector: np.ndarray,
):
    """
    Displacement field around a dislocation according to anisotropic elasticity theory
    described by [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

    Args:
        elastic_tensor ((3,3,3,3)-array): Elastic tensor
        positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
            coincides with the dislocation line.
        burgers_vector ((3,)-array): Burgers vector

    Returns:
        ((n, 3)-array): Displacement field (z-axis coincides with the dislocation line)
    """
    return Eshelby(elastic_tensor, burgers_vector).get_displacement(positions)


@units(
    outputs=lambda burgers_vector, positions: burgers_vector.u / positions.u
)
def get_dislocation_strain(
    elastic_tensor: np.ndarray,
    positions: np.ndarray,
    burgers_vector: np.ndarray,
):
    """
    Strain field around a dislocation according to anisotropic elasticity theory
    described by [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

    Args:
        elastic_tensor ((3,3,3,3)-array): Elastic tensor
        positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
            coincides with the dislocation line.
        burgers_vector ((3,)-array): Burgers vector

    Returns:
        ((n, 3, 3)-array): Strain field (z-axis coincides with the dislocation line)
    """
    return Eshelby(elastic_tensor, burgers_vector).get_strain(positions)


@units(
    outputs=lambda elastic_tensor, burgers_vector, positions: elastic_tensor.u
    * burgers_vector.u
    / positions.u
)
def get_dislocation_stress(
    elastic_tensor: np.ndarray,
    positions: np.ndarray,
    burgers_vector: np.ndarray,
):
    """
    Stress field around a dislocation according to anisotropic elasticity theory
    described by [Eshelby](https://doi.org/10.1016/0001-6160(53)90099-6).

    Args:
        elastic_tensor ((3,3,3,3)-array): Elastic tensor
        positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
            coincides with the dislocation line.
        burgers_vector ((3,)-array): Burgers vector

    Returns:
        ((n, 3, 3)-array): Stress field (z-axis coincides with the dislocation line)
    """
    strain = get_dislocation_strain(elastic_tensor, positions, burgers_vector)
    return np.einsum("ijkl,...kl->...ij", elastic_tensor, strain)


@units(
    outputs=lambda elastic_tensor, burgers_vector, positions: elastic_tensor.u
    * burgers_vector.u**2
    / positions.u**2
)
def get_dislocation_energy_density(
    elastic_tensor: np.ndarray,
    positions: np.ndarray,
    burgers_vector: np.ndarray,
):
    """
    Energy density field around a dislocation (product of stress and strain, cf. corresponding
    methods)

    Args:
        elastic_tensor ((3,3,3,3)-array): Elastic tensor
        positions ((n,2) or (n,3)-array): Position around a dislocation. The third axis
            coincides with the dislocation line.
        burgers_vector ((3,)-array): Burgers vector

    Returns:
        ((n,)-array): Energy density field
    """
    strain = get_dislocation_strain(elastic_tensor, positions, burgers_vector)
    return np.einsum("ijkl,...kl,...ij->...", elastic_tensor, strain, strain)


def get_dislocation_energy(
    elastic_tensor: np.ndarray,
    burgers_vector: np.ndarray,
    r_min: float,
    r_max: float,
    mesh: int = 100,
):
    """
    Energy per unit length along the dislocation line.

    Args:
        elastic_tensor ((3,3,3,3)-array): Elastic tensor
        burgers_vector ((3,)-array): Burgers vector
        r_min (float): Minimum distance from the dislocation core
        r_max (float): Maximum distance from the dislocation core
        mesh (int): Number of grid points for the numerical integration along the angle

    Returns:
        (float): Energy of dislocation per unit length

    The energy is defined by the product of the stress and strain (i.e. energy density),
    which is integrated over the plane vertical to the dislocation line. The energy density
    :math:`w` according to the linear elasticity is given by:

    .. math:
        w(r, \\theta) = A(\\theta)/r^2

    Therefore, the energy per unit length :math:`U` is given by:

    .. math:
        U = \\log(r_max/r_min)\\int A(\\theta)\\mathrm d\\theta

    This implies :math:`r_min` cannot be 0 as well as :math:`r_max` cannot be infinity. This
    is the consequence of the fact that the linear elasticity cannot describe the core
    structure properly, and a real medium is not infinitely large. While :math:`r_max` can
    be defined based on the real dislocation density, the choice of :math:`r_min` should be
    done carefully.
    """
    if r_min <= 0:
        raise ValueError("r_min must be a positive float")
    theta_range = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    r = np.stack((np.cos(theta_range), np.sin(theta_range)), axis=-1) * r_min
    strain = get_dislocation_strain(elastic_tensor, r, burgers_vector=burgers_vector)
    return (
        np.einsum("ijkl,nkl,nij->", elastic_tensor, strain, strain)
        / np.diff(theta_range)[0]
        * r_min**2
        * np.log(r_max / r_min)
    )


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
