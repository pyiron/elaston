# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import string
from typing import Annotated

import numpy as np
from semantikon.converter import units

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


def normalize(x: np.ndarray) -> np.ndarray:
    """
    Normalize a vector or an array of vectors.

    Args:
        x (numpy.ndarray): A vector or an array of vectors.

    Returns:
        numpy.ndarray: Normalized vector or array of vectors.
    """
    return (x.T / np.linalg.norm(x, axis=-1).T).T


def orthonormalize(vectors: list) -> np.ndarray:
    """
    Orthonormalize a set of vectors.

    Args:
        vectors (list): A list of vectors.

    Returns:
        numpy.ndarray: An orthonormal basis.
    """
    if np.shape(vectors) == (3, 3) and np.linalg.det(vectors) <= 0:
        raise ValueError("Vectors not independent or not right-handed")
    x = np.eye(3)
    x[:2] = normalize(np.asarray(vectors)[:2])
    x[1] = x[1] - np.einsum("i,i,j->j", x[0], x[1], x[0])
    x[2] = np.cross(x[0], x[1])
    if np.isclose(np.linalg.det(x), 0):
        raise ValueError("Vectors not independent")
    return normalize(x)


def get_plane(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Get a plane perpendicular to a vector.

    Args:
        T (numpy.ndarray): A vector.

    Returns:
        tuple: A pair of vectors that span the plane.
    """
    x = normalize(np.random.random(T.shape))
    x = normalize(x - np.einsum("...i,...i,...j->...j", x, T, T))
    y = np.cross(T, x)
    return x, y


def index_from_voigt(i: int, j: int) -> int:
    """
    Convert Voigt notation to matrix index.

    Args:
        i (int): Voigt index.
        j (int): Voigt index.

    Returns:
        int: Matrix index.
    """
    if i == j:
        return i
    else:
        return 6 - i - j


@units
def C_from_voigt(
    C_in: Annotated[np.ndarray, {"units": "=C"}], inverse: bool = False
) -> Annotated[np.ndarray, {"units": "=C"}]:
    """
    Convert elastic tensor in Voigt notation to matrix notation.

    Args:
        C_in (numpy.ndarray): Elastic tensor in Voigt notation.
        inverse (bool): Whether to use the inverse Voigt notation.

    Returns:
        numpy.ndarray: Elastic tensor in matrix notation.
    """
    C_v = np.array(C_in)
    if inverse:
        C_v[3:] /= 2
        C_v[:, 3:] /= 2
    C = np.zeros((3, 3, 3, 3))
    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                for ll in range(3):
                    C[ii, jj, kk, ll] = C_v[
                        index_from_voigt(ii, jj), index_from_voigt(kk, ll)
                    ]
    return C


@units
def C_to_voigt(
    C_in: Annotated[np.ndarray, {"units": "=C"}],
) -> Annotated[np.ndarray, {"units": "=C"}]:
    """
    Convert elastic tensor in matrix notation to Voigt notation.

    Args:
        C_in (numpy.ndarray): Elastic tensor in matrix notation.

    Returns:
        numpy.ndarray: Elastic tensor in Voigt notation.
    """
    if np.shape(C_in) == (6, 6):
        return np.asarray(C_in)
    C = np.zeros((6, 6))
    for ii in range(3):
        for jj in range(ii + 1):
            for kk in range(3):
                for ll in range(kk + 1):
                    C[index_from_voigt(ii, jj), index_from_voigt(kk, ll)] = C_in[
                        ii, jj, kk, ll
                    ]
    return C


@units
def voigt_average(
    C_11: Annotated[float, {"units": "=C"}],
    C_12: Annotated[float, {"units": "=C"}],
    C_44: Annotated[float, {"units": "=C"}],
) -> Annotated[np.ndarray, {"units": "=C"}]:
    """Make isotropic elastic tensor from C_11, C_12, and C_44."""
    return np.array([[3, 2, 4], [1, 4, -2], [1, -1, 3]]) / 5 @ [C_11, C_12, C_44]


def _get_einsum_str(
    shape: tuple[int, ...], axes: np.ndarray, inverse: bool = True
) -> str:
    """
    Get the einsum string for the given shape.

    Args:
        shape (tuple): Shape of the tensor.
        axes (numpy.ndarray): Axes to rotate.
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


def crystal_to_box(
    tensor: np.ndarray, orientation: np.ndarray, axes: np.ndarray | None = None
) -> np.ndarray:
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


def box_to_crystal(
    tensor: np.ndarray, orientation: np.ndarray, axes: np.ndarray | None = None
) -> np.ndarray:
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


def _rotate_tensor(
    tensor: np.ndarray,
    orientation: np.ndarray,
    inverse: bool,
    axes: np.ndarray | None = None,
) -> np.ndarray:
    v = np.atleast_2d(tensor)
    if axes is None:
        axes = np.where(np.array(v.shape) == 3)[0]
    axes = np.atleast_1d(axes)
    return np.einsum(
        _get_einsum_str(v.shape, axes=axes, inverse=inverse),
        *len(axes) * [orthonormalize(orientation)],
        v,
    ).reshape(tensor.shape)


@units
def get_compliance_tensor(
    elastic_tensor: Annotated[np.ndarray, {"units": "=C"}],
    voigt: bool = False,
) -> Annotated[np.ndarray, {"units": "=1/C"}]:
    S = np.linalg.inv(elastic_tensor)
    if voigt:
        return S
    return C_from_voigt(S, inverse=True)
