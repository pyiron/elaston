# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Optional

import numpy as np
from semantikon.converter import units
from semantikon.typing import u

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
    C_23: Optional[float] = None,
    C_33: Optional[float] = None,
    C_44: Optional[float] = None,
    C_55: Optional[float] = None,
    C_66: Optional[float] = None,
):
    """
    Get the elastic tensor from the elastic constants

    Args:
        C_tensor (np.ndarray): Elastic tensor in Voigt notation or full tensor
        C_11 (float): Elastic constant
        C_12 (float): Elastic constant
        C_13 (float): Elastic constant
        C_22 (float): Elastic constant
        C_23 (float): Elastic constant
        C_33 (float): Elastic constant
        C_44 (float): Elastic constant
        C_55 (float): Elastic constant
        C_66 (float): Elastic constant

    Returns:
        np.ndarray: Elastic tensor in Voigt notation
    """
    if C_tensor is not None:
        if np.shape(C_tensor) in [(6, 6), (3, 3, 3, 3)]:
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
    if C_23 is None:
        C_23 = C_12
    if C_22 is None:
        C_22 = C_11
    if C_33 is None:
        C_33 = C_11
    if C_55 is None:
        C_55 = C_44
    if C_66 is None:
        C_66 = C_44
    C = _convert_elastic_constants(C_11, C_12, C_13, C_22, C_23, C_33, C_44, C_55, C_66)
    return C


@units
def _convert_elastic_constants(
    C_11: u(float, units="=A"),
    C_12: u(float, units="=A"),
    C_13: u(float, units="=A"),
    C_22: u(float, units="=A"),
    C_23: u(float, units="=A"),
    C_33: u(float, units="=A"),
    C_44: u(float, units="=A"),
    C_55: u(float, units="=A"),
    C_66: u(float, units="=A"),
) -> u(np.ndarray, units="=A"):
    return np.array(
        [
            [C_11, C_12, C_13, 0, 0, 0],
            [C_12, C_22, C_23, 0, 0, 0],
            [C_13, C_23, C_33, 0, 0, 0],
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
        np.ndarray: Elastic tensor in Voigt notation
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
    return _convert_elastic_moduli(E, nu, mu)


@units
def _convert_elastic_moduli(
    E: u(float, units="=A"),
    nu: u(float, units="=A"),
    mu: u(float, units="=A"),
) -> u(np.ndarray, units="=A"):
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


@units
def _get_reuss_average_values(
    C: u(np.ndarray, units="=A"),
) -> u(np.ndarray, units="=A"):
    S = np.linalg.inv(C)
    S[3:, 3:] /= 4
    S = get_voigt_average(S)
    S = get_elastic_tensor_from_tensor(**S)
    C = np.linalg.inv(S)
    return C


def get_reuss_average(C):
    """
    Get the Reuss average of the elastic constants

    Args:
        C (np.ndarray): Elastic constants

    Returns:
        dict: Reuss average
    """
    C = _get_reuss_average_values(C)
    return dict(zip(["C_11", "C_12", "C_44"], [C[0, 0], C[0, 1], C[3, 3] / 4]))


def is_cubic(C):
    """
    Check if the material is cubic

    Args:
        C (np.ndarray): Elastic constants in Voigt notation or full tensor

    Returns:
        bool: True if the material is cubic
    """
    if np.shape(C) == (3, 3, 3, 3):
        C = tools.C_to_voigt(C)
    return all(
        [
            np.isclose(np.ptp(C[ind]), 0)
            for ind in [get_C_11_indices(), get_C_12_indices(), get_C_44_indices()]
        ]
    )


def is_isotropic(C):
    """
    Check if the material is isotropic

    Args:
        C (np.ndarray): Elastic constants in Voigt notation or full tensor

    Returns:
        bool: True if the material is isotropic
    """
    if np.shape(C) == (3, 3, 3, 3):
        C = tools.C_to_voigt(C)
    return is_cubic(C) and np.isclose(get_zener_ratio(C), 1)


def get_zener_ratio(C):
    """
    Get the Zener anisotropy ratio

    Args:
        C (np.ndarray): Elastic constants in Voigt notation

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
    if np.shape(C) == (3, 3, 3, 3):
        C = tools.C_to_voigt(C)
    if not is_isotropic(C):
        raise ValueError("The material must be isotropic")
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


def initialize_elastic_tensor(
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
    """
    Initialize the elastic tensor

    Args:
        C_tensor (np.ndarray): Elastic tensor in Voigt notation or full
            tensor
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

    Returns:
        np.ndarray: Elastic tensor in Voigt notation

    You can define either the full elastic tensor via C_tensor, some
    components of the elastic tensor or the elastic moduli. If you
    define the elastic moduli, the elastic tensor will be calculated
    from them. When two components of the elastic tensor are given, the
    resulting tensor will be isotropic. If at least three components are
    given, there must be at least C_11, C_12, and C_44.
    """
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
        return get_elastic_tensor_from_tensor(
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
        return get_elastic_tensor_from_moduli(
            E=youngs_modulus,
            nu=poissons_ratio,
            mu=shear_modulus,
        )
