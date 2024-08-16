# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
import string

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


def normalize(x):
    return (x.T / np.linalg.norm(x, axis=-1).T).T


def orthonormalize(vectors):
    x = np.eye(3)
    x[:2] = normalize(np.asarray(vectors)[:2])
    x[1] = x[1] - np.einsum("i,i,j->j", x[0], x[1], x[0])
    x[2] = np.cross(x[0], x[1])
    if np.isclose(np.linalg.det(x), 0):
        raise ValueError("Vectors not independent")
    return normalize(x)


def get_plane(T):
    x = normalize(np.random.random(T.shape))
    x = normalize(x - np.einsum("...i,...i,...j->...j", x, T, T))
    y = np.cross(T, x)
    return x, y


def index_from_voigt(i, j):
    if i == j:
        return i
    else:
        return 6 - i - j


def C_from_voigt(C_in):
    C = np.zeros((3, 3, 3, 3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i, j, k, l] = C_in[index_from_voigt(i, j), index_from_voigt(k, l)]
    return C


def C_to_voigt(C_in):
    C = np.zeros((6, 6))
    for i in range(3):
        for j in range(i + 1):
            for k in range(3):
                for l in range(k + 1):
                    C[index_from_voigt(i, j), index_from_voigt(k, l)] = C_in[i, j, k, l]
    return C


def coeff_to_voigt(C_in):
    C = np.zeros((6, 6))
    C[:3, :3] = C_in[1]
    C[:3, :3] += np.eye(3) * (C_in[0] - C_in[1])
    C[3:, 3:] = C_in[2] * np.eye(3)
    return C


def voigt_average(C_11: float, C_12: float, C_44: float):
    """Make isotropic elastic tensor from C_11, C_12, and C_44."""
    mat = np.array([[0.6, 0.4, 0.8], [0.2, 0.8, -0.4], [0.2, -0.2, 0.6]])
    return mat @ [C_11, C_12, C_44]


def _get_einsum_str(shape: tuple, inverse: bool = True, axes=None) -> str:
    """
    Get the einsum string for the given shape.

    Args:
        shape (tuple): Shape of the tensor.
        inverse (bool): Whether to use the inverse einsum string.

    Returns:
        str: Einsum string.

    """
    s = [string.ascii_lowercase[i] for i in range(len(shape))]
    s_rot = ""
    s_mul = ""
    for ii, ss in enumerate(s):
        if ii in axes:
            if inverse:
                s_rot += ss.upper() + ss + ","
            else:
                s_rot += ss + ss.upper() + ","
            s_mul += ss.upper()
        else:
            s_mul += ss
    return s_rot + s_mul + "->" + "".join(s)


def crystal_to_box(tensor, orientation, axes=None):
    """
    Translate a tensor given in the crystal coordinate system to the box
    coordinate system. Crystal coordinates are (usually) given by [[1, 0, 0],
    [0, 1, 0], [0, 0, 1]] and box coordinates are given by the lattice
    vectors, such as [[1, 1, 1], [1, -1, 0], [1, -2, 1]].

    Args:
        tensor (np.ndarray): Tensor to be rotated.
        orientation (np.ndarray): Orientation matrix.
        axes (np.ndarray): Axes to rotate.
    """
    return _rotate_tensor(tensor, orientation, inverse=False, axes=axes)


def box_to_crystal(tensor, orientation, axes=None):
    """
    Translate a tensor given in the box coordinate system to the crystal
    coordinate system. Crystal coordinates are (usually) given by [[1, 0, 0],
    [0, 1, 0], [0, 0, 1]] and box coordinates are given by the lattice
    vectors, such as [[1, 1, 1], [1, -1, 0], [1, -2, 1]].

    Args:
        tensor (np.ndarray): Tensor to be rotated.
        orientation (np.ndarray): Orientation matrix.
        axes (np.ndarray): Axes to rotate.
    """
    return _rotate_tensor(tensor, orientation, inverse=True, axes=axes)

def _rotate_tensor(tensor, orientation, inverse, axes=None):
    v = np.atleast_2d(tensor)
    if axes is None:
        axes = np.where(np.array(v.shape) == 3)[0]
    axes = np.atleast_1d(axes)
    return np.einsum(
        _get_einsum_str(v.shape, inverse=inverse, axes=axes),
        *len(axes) * [orthonormalize(orientation)],
        v
    ).reshape(tensor.shape)
