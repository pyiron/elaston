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


def check_is_tensor(**kwargs):
    d = {k: v for k, v in kwargs.items() if v is not None}
    if len(d) < 2:
        raise ValueError("At least two of the elastic constants must be given")
    if any([k.startswith("C_") for k in d.keys()]):
        if any([not k.startswith("C_") for k in d.keys()]):
            raise ValueError(
                "Either elastic constants or Young's modulus and Poisson's ratio"
                " must be given but not both"
            )
        return True
    return False


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
        is_tensor = check_consistency(
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
