# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from pint import Quantity, Unit
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


def _get_ureg(args, kwargs):
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, Quantity):
            return arg._REGISTRY
    return None


def _get_input(kwargs, inputs):
    kwargs_tmp = kwargs.copy()
    for key, val in kwargs_tmp.items():
        if isinstance(val, Quantity):
            if inputs is not None and key in inputs:
                kwargs_tmp[key] = val.to(inputs[key]).magnitude
            else:
                kwargs_tmp[key] = val.magnitude
    return kwargs_tmp


def _get_output_units(outputs, kwargs, ureg):
    def f(out, kwargs=kwargs, ureg=ureg):
        return out(**kwargs) if callable(out) else getattr(ureg, out)

    try:
        if callable(outputs) or isinstance(outputs, str):
            return f(outputs)
        if isinstance(outputs, (list, tuple)):
            return tuple([f(output) for output in outputs])
    except AttributeError as e:
        raise SyntaxError(
            "This function return an output with a relative unit. Either you"
            f" define all the units or none of them: {e}"
        )


def _check_inputs_and_outputs(inp, out):
    assert inp is None or isinstance(inp, dict)
    assert out is None or callable(out) or isinstance(out, (list, tuple, str))
    if inp is not None:
        if callable(out) or (
            isinstance(out, (list, tuple)) and any(map(callable, out))
        ):
            raise ValueError(
                "You cannot use relative output units when inpput units are defined"
            )


def units(outputs=None, inputs=None):
    _check_inputs_and_outputs(inputs, outputs)

    def decorator(func):
        def wrapper(*args, **kwargs):
            ureg = _get_ureg(args, kwargs)
            if ureg is None:
                return func(*args, **kwargs)
            # This step unifies args and kwargs
            kwargs.update(zip(getfullargspec(func).args, args))
            if outputs is not None:
                output_units = _get_output_units(outputs, kwargs, ureg)
            result = func(**_get_input(kwargs, inputs))
            if outputs is not None:
                if isinstance(output_units, Unit):
                    return result * output_units
                else:
                    return tuple([res * out for res, out in zip(result, output_units)])
            return result

        return wrapper

    return decorator
