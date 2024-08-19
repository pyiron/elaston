# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pint import Quantity
from inspect import getfullargspec

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


def _is_plain(inputs, outputs, args, kwargs):
    if inputs is None and outputs is None:
        return True
    if any([isinstance(arg, Quantity) for arg in args]):
        return False
    if any([isinstance(val, Quantity) for val in kwargs.values()]):
        return False
    return True


def _get_input(kwargs, inputs):
    kwargs_tmp = kwargs.copy()
    for key, val in kwargs_tmp.items():
        if isinstance(val, Quantity):
            if key in inputs:
                kwargs_tmp[key] = val.to(inputs[key]).magnitude
            else:
                kwargs_tmp[key] = val.magnitude
    return kwargs_tmp


def units(outputs=None, inputs=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if _is_plain(inputs, outputs, args, kwargs):
                return func(*args, **kwargs)
            kwargs.update(zip(getfullargspec(func).args, args))
            result = func(**_get_input(kwargs, inputs))
            return result
        return wrapper
    return decorator
