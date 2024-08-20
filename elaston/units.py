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
    """
    Get the registry from the arguments.

    Parameters
    ----------
    args : tuple
    kwargs : dict

    Returns
    -------
    pint.UnitRegistry
    """
    for arg in args + tuple(kwargs.values()):
        if isinstance(arg, Quantity):
            return arg._REGISTRY
    return None


def _pint_to_value(kwargs, inputs):
    """
    Convert pint quantities to values.

    Parameters
    ----------
    kwargs : dict
    inputs : dict

    Returns
    -------
    dict
    """
    kwargs_tmp = kwargs.copy()
    for key, val in kwargs_tmp.items():
        if isinstance(val, Quantity):
            if inputs is not None and key in inputs:
                kwargs_tmp[key] = val.to(inputs[key]).magnitude
            else:
                kwargs_tmp[key] = val.magnitude
    return kwargs_tmp


def _get_output_units(outputs, kwargs, ureg):
    """
    Get the output units.

    Parameters
    ----------
    outputs : str, list, tuple, callable
    kwargs : dict
    ureg : pint.UnitRegistry

    Returns
    -------
    pint.Unit, tuple
    """
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
    """
    Check the inputs and outputs.

    Parameters
    ----------
    inp : dict
    out : str, list, tuple, callable
    """
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
    """
    Decorator to handle units in functions.

    Parameters
    ----------
    outputs : str, list, tuple, callable, optional
        Output units. If a string, it should be a valid unit. If a list or
        tuple, it should contain valid units. If a callable, it should return a
        valid unit.
    inputs : dict, optional

    Returns
    -------
    callable
    """
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
            result = func(**_pint_to_value(kwargs, inputs))
            if outputs is not None:
                if isinstance(output_units, Unit):
                    return result * output_units
                else:
                    return tuple([res * out for res, out in zip(result, output_units)])
            return result

        return wrapper

    return decorator
