# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from typing import Optional
from elaston import tools

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


# ref https://en.m.wikiversity.org/wiki/Elasticity/Constitutive_relations


def get_C_11_indices():
    return [0, 1, 2], [0, 1, 2]


def get_C_44_indices():
    return [3, 4, 5], [3, 4, 5]


def get_C_12_indices():
    return [0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]


def check_is_tensor(**kwargs):
    """
    Check if the elastic constants are given as a tensor or as Young's modulus,
    Poisson's ratio, and/or shear modulus

    Args:
        **kwargs: elastic constants or Young's modulus, Poisson's ratio,
            and/or shear modulus
    """
    d = {k: v for k, v in kwargs.items() if v is not None}
    if len(d) < 2 and "C_tensor" not in d.keys():
        raise ValueError("At least two of the elastic constants must be given")
    if any([k.startswith("C_") for k in d.keys()]):
        if any([not k.startswith("C_") for k in d.keys()]):
            raise ValueError(
                "Either elastic constants or Young's modulus, Poisson's ratio"
                " and/or shear modulus must be given but not both"
            )
        return True
    return False


def get_elastic_tensor_from_tensor(
    C_tensor: Optional[np.ndarray] = None,
    C_11: Optional[float] = None,
    C_12: Optional[float] = None,
    C_13: Optional[float] = None,
    C_22: Optional[float] = None,
    C_33: Optional[float] = None,
    C_44: Optional[float] = None,
    C_55: Optional[float] = None,
    C_66: Optional[float] = None,
):
    if C_tensor is not None:
        if np.shape(C_tensor) == (6, 6):
            return np.asarray(C_tensor)
        elif np.shape(C_tensor) == (3, 3, 3, 3):
            return tools.C_to_voigt(C_tensor)
        else:
            raise ValueError(
                f"Invalid shape of the elastic tensor: {np.shape(C_tensor)}"
            )
    if C_11 is None and C_12 is not None and C_44 is not None:
        C_11 = C_12 + 2 * C_44
    elif C_11 is not None and C_12 is None and C_44 is not None:
        C_12 = C_11 - 2 * C_44
    elif C_11 is not None and C_12 is not None and C_44 is None:
        C_44 = (C_11 - C_12) / 2
    elif C_11 is None or C_12 is None or C_44 is None:
        raise ValueError("Out of C_11, C_12, and C_44 at least two must be given")
    if C_13 is None:
        C_13 = C_12
    if C_22 is None:
        C_22 = C_11
    if C_33 is None:
        C_33 = C_11
    if C_55 is None:
        C_55 = C_44
    if C_66 is None:
        C_66 = C_44
    return np.array(
        [
            [C_11, C_12, C_13, 0, 0, 0],
            [C_12, C_22, C_13, 0, 0, 0],
            [C_13, C_13, C_33, 0, 0, 0],
            [0, 0, 0, C_44, 0, 0],
            [0, 0, 0, 0, C_55, 0],
            [0, 0, 0, 0, 0, C_66],
        ]
    )


def get_elastic_tensor_from_moduli(
    E: Optional[float] = None,
    nu: Optional[float] = None,
    mu: Optional[float] = None,
):
    """
    Get the elastic tensor from Young's modulus, Poisson's ratio, and/or shear modulus

    Args:
        E (float): Young's modulus
        nu (float): Poisson's ratio
        mu (float): shear modulus

    Returns:
        np.ndarray: Elastic tensor
    """
    if E is None and nu is not None and mu is not None:
        E = 2 * mu * (1 + nu)
    elif E is not None and nu is None and mu is not None:
        nu = E / (2 * mu) - 1
    elif E is not None and nu is not None and mu is None:
        mu = E / (2 * (1 + nu))
    elif E is None or nu is None or mu is None:
        raise ValueError(
            "Out of Young's modulus, Poisson's ratio, and shear modulus"
            " at least two must be given"
        )
    return np.linalg.inv(
        [
            [1 / E, -nu / E, -nu / E, 0, 0, 0],
            [-nu / E, 1 / E, -nu / E, 0, 0, 0],
            [-nu / E, -nu / E, 1 / E, 0, 0, 0],
            [0, 0, 0, 1 / mu, 0, 0],
            [0, 0, 0, 0, 1 / mu, 0],
            [0, 0, 0, 0, 0, 1 / mu],
        ]
    )


def get_voigt_average(C):
    """
    Get the Voigt average of the elastic constants

    Args:
        C (np.ndarray): Elastic constants

    Returns:
        dict: Voigt average
    """
    C_11 = np.mean(C[get_C_11_indices()])
    C_12 = np.mean(C[get_C_12_indices()])
    C_44 = np.mean(C[get_C_44_indices()])
    return dict(zip(["C_11", "C_12", "C_44"], tools.voigt_average(C_11, C_12, C_44)))


def get_reuss_average(C):
    """
    Get the Reuss average of the elastic constants

    Args:
        C (np.ndarray): Elastic constants

    Returns:
        dict: Reuss average
    """
    S = np.linalg.inv(C)
    S[3:, 3:] /= 4
    S = get_voigt_average(S)
    S = get_elastic_tensor_from_tensor(**S)
    C = np.linalg.inv(S)
    return dict(zip(["C_11", "C_12", "C_44"], [C[0, 0], C[0, 1], C[3, 3] / 4]))


def is_cubic(C):
    """
    Check if the material is cubic

    Args:
        C (np.ndarray): Elastic constants

    Returns:
        bool: True if the material is cubic
    """
    return all(
        [
            np.isclose(np.ptp(C[ind]), 0)
            for ind in [get_C_11_indices(), get_C_12_indices(), get_C_44_indices()]
        ]
    )


def get_zener_ratio(C):
    """
    Get the Zener anisotropy ratio

    Args:
        C (np.ndarray): Elastic constants

    Returns:
        float: Zener anisotropy ratio
    """
    if not is_cubic(C):
        raise ValueError("The material must be cubic")
    C_11 = np.mean(C[get_C_11_indices()])
    C_12 = np.mean(C[get_C_12_indices()])
    C_44 = np.mean(C[get_C_44_indices()])
    return 2 * C_44 / (C_11 - C_12)


def get_unique_elastic_constants(C):
    indices = np.sort(np.unique(np.round(C, decimals=8), return_index=True)[1])
    i, j = np.unravel_index(indices, (6, 6))
    return {
        f"C_{ii + 1}{jj + 1}": CC
        for ii, jj, CC in zip(i, j, C.flatten()[indices])
        if not np.isclose(CC, 0)
    }


def get_elastic_moduli(C):
    C_11 = np.mean(C[get_C_11_indices()])
    C_12 = np.mean(C[get_C_12_indices()])
    C_44 = np.mean(C[get_C_44_indices()])
    return {
        "youngs_modulus": (C_11 - C_12) * (C_11 + 2 * C_12) / (C_11 + C_12),
        "poissons_ratio": C_12 / (C_11 + C_12),
        "shear_modulus": C_44,
        "lamé_first_parameter": (C_12 + C_44) / 2,
        "bulk_modulus": (C_11 + 2 * C_12) / 3,
    }



class ElasticConstants:
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
    ):
        is_tensor = check_is_tensor(
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
        if is_tensor:
            self._elastic_tensor = get_elastic_tensor_from_tensor(
                C_tensor=C_tensor,
                C_11=C_11,
                C_12=C_12,
                C_13=C_13,
                C_22=C_22,
                C_33=C_33,
                C_44=C_44,
                C_55=C_55,
                C_66=C_66,
            )
        else:
            self._elastic_tensor = get_elastic_tensor_from_moduli(
                E=youngs_modulus,
                nu=poissons_ratio,
                mu=shear_modulus,
            )

    @property
    def elastic_tensor(self):
        return self._elastic_tensor

    def get_voigt_average(self):
        return ElasticConstants(**get_voigt_average(self.elastic_tensor))

    def get_reuss_average(self):
        return ElasticConstants(**get_reuss_average(self.elastic_tensor))

    def is_cubic(self):
        return is_cubic(self.elastic_tensor)

    def is_isotropic(self):
        return self.is_cubic and np.isclose(self.get_zener_ratio(), 1)

    def get_zener_ratio(self):
        return get_zener_ratio(self.elastic_tensor)

    def get_unique_elastic_constants(self):
        return get_unique_elastic_constants(self.elastic_tensor)

    def get_elastic_moduli(self):
        if not self.is_isotropic():
            raise ValueError(
                "The material must be isotropic. Re-instantiate with isotropic"
                " elastic constants or run an averaging method"
                " (get_voigt_average, get_reuss_average) first"
            )
        return get_elastic_moduli(self.elastic_tensor)
