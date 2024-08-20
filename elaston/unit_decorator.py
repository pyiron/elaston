# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import re
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
            if inputs is not None and key in inputs:
                kwargs_tmp[key] = val.to(inputs[key]).magnitude
            else:
                kwargs_tmp[key] = val.magnitude
    return kwargs_tmp


def replace_vars(expression, variables):
    return re.sub(
        r"\b[a-zA-Z_]\w*\b",
        lambda match: (
            f'kwargs["{match.group(0)}"].u'
            if match.group(0) in variables
            else match.group(0)
        ),
        expression,
    )


def _get_output_units(outputs, kwargs):
    try:
        if callable(outputs):
            return outputs(**kwargs)
        if isinstance(outputs, (list, tuple)):
            return tuple([output(**kwargs) for output in outputs])
    except AttributeError:
        raise SyntaxError(
            f"Invalid syntax: {outputs} for the given variables {kwargs} (probably"
            " partly missing units)"
        )


def units(outputs=None, inputs=None):
    if inputs is not None and outputs is not None:
        raise ValueError(
            "You can only specify either inputs or outputs, not both."
        )

    def decorator(func):
        def wrapper(*args, **kwargs):
            if _is_plain(inputs, outputs, args, kwargs):
                return func(*args, **kwargs)
            # This step unifies args and kwargs
            kwargs.update(zip(getfullargspec(func).args, args))
            if outputs is not None:
                output_units = _get_output_units(outputs, kwargs)
            result = func(**_get_input(kwargs, inputs))
            if outputs is not None:
                if callable(outputs):
                    return result * output_units
                else:
                    return tuple(
                        [
                            res * out for res, out in zip(result, output_units)
                        ]
                    )
            return result

        return wrapper

    return decorator
