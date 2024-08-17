# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from functools import cached_property

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
        assert elastic_tensor.shape == (3, 3, 3, 3)
        assert burgers_vector.shape == (3,)
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
