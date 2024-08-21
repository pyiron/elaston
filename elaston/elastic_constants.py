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
    if len(d) < 2:
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
    C_11: Optional[float] = None,
    C_12: Optional[float] = None,
    C_13: Optional[float] = None,
    C_22: Optional[float] = None,
    C_33: Optional[float] = None,
    C_44: Optional[float] = None,
    C_55: Optional[float] = None,
    C_66: Optional[float] = None,
):
    if C_11 is None and C_12 is not None and C_44 is not None:
        C_11 = C_12 + 2 * C_44
    elif C_11 is not None and C_12 is None and C_44 is not None:
        C_12 = C_11 - 2 * C_44
    elif C_11 is not None and C_12 is not None and C_44 is None:
        C_44 = (C_11 - C_12) / 2
    else:
        raise ValueError(
            "Out of C_11, C_12, and C_44 at least two must be given"
        )
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


def get_elastic_tensor_from_properties(
    E: Optional[float] = None,
    nu: Optional[float] = None,
    mu: Optional[float] = None,
):
    if E is None and nu is not None and mu is not None:
        E = 2 * mu * (1 + nu)
    elif E is not None and nu is None and mu is not None:
        nu = E / (2 * mu) - 1
    elif E is not None and nu is not None and mu is None:
        mu = E / (2 * (1 + nu))
    else:
        raise ValueError(
            "Out of Young's modulus, Poisson's ratio, and shear modulus"
            " at least two must be given"
        )
    return np.linalg.inv(
        [
            [1 / E, -nu / E, -nu / E, 0, 0, 0],
            [-nu / E, 1 / E, -nu / E, 0, 0, 0],
            [-nu / E, -nu / E, 1 / E, 0, 0, 0],
            [0, 0, 0, 1 / (2 * mu), 0, 0],
            [0, 0, 0, 0, 1 / (2 * mu), 0],
            [0, 0, 0, 0, 0, 1 / (2 * mu)],
        ]
    )


class ElasticConstants:
    def __init__(
        self,
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
            self._elastic_tensor = get_elastic_tensor_from_properties(
                E=youngs_modulus,
                nu=poissons_ratio,
                mu=shear_modulus,
            )

    @property
    def elastic_tensor(self):
        return self._elastic_tensor

    def get_voigt_average(self):
        C_11 = np.mean(self.elastic_tensor[get_C_11_indices()])
        C_12 = np.mean(self.elastic_tensor[get_C_12_indices()])
        C_44 = np.mean(self.elastic_tensor[get_C_44_indices()])
        return tools.voigt_average([C_11, C_12, C_44])
