# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

from abc import ABC
from functools import wraps
import inspect
from typing import Annotated, Any, ClassVar, get_type_hints
import warnings

from pint import Quantity, Unit
from pyiron_snippets.factory import classfactory, sanitize_callable_name

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


def extend_callable_with_kwargs(f):
    # Retrieve the signature of the callable `f`
    sig = inspect.signature(f)

    def wrapper(*args, **kwargs):
        # Filter `kwargs` to include only the parameters that `f` accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        # Call `f` with the positional arguments and filtered keyword arguments
        return f(*args, **filtered_kwargs)

    return wrapper


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
        if callable(out):
            return extend_callable_with_kwargs(out)(**kwargs)
        else:
            return getattr(ureg, out)

    try:
        if callable(outputs) or isinstance(outputs, str):
            return f(outputs)
        if isinstance(outputs, (list, tuple)):
            return tuple([f(output) for output in outputs])
    except AttributeError as e:
        warnings.warn(
            "This function return an output with a relative unit. Either you"
            f" define all the units or none of them: {e}",
            SyntaxWarning,
        )
        return None


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
                "You cannot use relative output units when input units are defined"
            )


def _get_input_args(func, *args, **kwargs):
    signature = inspect.signature(func)
    bound_args = signature.bind_partial(*args, **kwargs)
    bound_args.apply_defaults()  # Fill in the default values
    bound_args = dict(bound_args.arguments)
    return bound_args


def get_units_from_type_hints(func):
    return {
        key: value.__metadata__[0]
        for key, value in get_type_hints(func, include_extras=True).items()
        if hasattr(value, "__metadata__")
    }


def get_input_units_from_type_hints(func):
    d = get_units_from_type_hints(func)
    if "return" in d:
        del d["return"]
    if len(d) == 0:
        return None
    return d


def get_output_units_from_type_hints(func):
    return get_units_from_type_hints(func).get("return", None)


def units(func=None, *, outputs=None, inputs=None):
    # Perform initial checks
    _check_inputs_and_outputs(inputs, outputs)
    # If func is None, this means the decorator is called with parentheses
    if func is None:
        # Return the actual decorator that expects the function
        def decorator(func):
            return _units_decorator(func, inputs, outputs)

        return decorator
    else:
        # The decorator is called without parentheses, so func is the actual function
        return _units_decorator(func, inputs, outputs)


def _units_decorator(func, inputs, outputs):

    # If inputs or outputs are None, set them based on the function signature
    if inputs is None:
        inputs = get_input_units_from_type_hints(func)
    if outputs is None:
        outputs = get_output_units_from_type_hints(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        ureg = _get_ureg(args, kwargs)
        if ureg is None:
            return func(*args, **kwargs)
        bound_args = _get_input_args(func, *args, **kwargs)
        if outputs is not None:
            output_units = _get_output_units(outputs, bound_args, ureg)
        result = func(**_pint_to_value(bound_args, inputs))
        if outputs is not None and output_units is not None:
            if isinstance(output_units, Unit):
                return result * output_units
            else:
                return tuple([res * out for res, out in zip(result, output_units)])
        return result

    return wrapper


def optional_units(*args):
    for arg in args:
        if isinstance(arg, Quantity):
            return arg.u
    return 1


class HasUnits(ABC):
    _hint: ClassVar[Any]

    def __class_getitem__(cls, metadata):
        return Annotated[cls._hint, metadata]

    def __getitem__(self, unit):
        # The subscript method for handling unit strings
        if not isinstance(unit, str):
            raise TypeError("Unit must be a string.")
        # Return a new instance or perform any specific behavior you want
        return self.base_type_with_unit(unit)

    def base_type_with_unit(self, unit):
        # Placeholder for an actual implementation.
        # This could, for example, return a custom type or a decorated function
        # that incorporates the unit information.
        return f"{self.base_type.__name__} with unit '{unit}'"



@classfactory
def u(hint, /):
    return (
        sanitize_callable_name(str(hint)),
        (HasUnits,),
        {"_hint": hint},
        {},
    )
