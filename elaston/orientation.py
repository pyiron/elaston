# coding: utf-8
# Copyright (c) Max-Planck-Institut für Eisenforschung GmbH - Computational Materials Design (CM) Department
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np

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


def get_dislocation_orientation(
    dislocation_type: str = "screw", crystal: str = "bcc"
) -> dict[str, np.ndarray]:
    """
    Get the orientation of a dislocation in a crystal.

    Args:
        dislocation_type (str): Type of dislocation. "screw" or "edge".
        crystal (str): Crystal structure. "bcc" or "fcc".

    Returns:
        dict: Dictionary containing the following keys:
            - glide_plane (np.ndarray): Glide plane normal.
            - burgers_vector (np.ndarray): Burgers vector.
            - dislocation_line (np.ndarray): Dislocation line.
            - orientation (np.ndarray): Dislocation orientation matrix; First
                row is the glide plane, and the last row is the dislocation line.
    """
    assert dislocation_type in ["screw", "edge"]
    assert crystal in ["bcc", "fcc"]
    result: dict[str, np.ndarray] = {}
    if crystal == "bcc":
        if dislocation_type == "screw":
            result = {
                "glide_plane": np.array([1, -1, 0]),
                "burgers_vector": np.array([1, 1, 1]),
                "dislocation_line": np.array([1, 1, 1]),
            }
        elif dislocation_type == "edge":
            result = {
                "glide_plane": np.array([1, -1, 0]),
                "burgers_vector": np.array([1, 1, 1]),
                "dislocation_line": np.array([1, 1, -2]),
            }
    elif crystal == "fcc":
        if dislocation_type == "screw":
            result = {
                "glide_plane": np.array([1, 1, 1]),
                "burgers_vector": np.array([1, -1, 0]),
                "dislocation_line": np.array([1, -1, 0]),
            }
        elif dislocation_type == "edge":
            result = {
                "glide_plane": np.array([1, 1, 1]),
                "burgers_vector": np.array([1, -1, 0]),
                "dislocation_line": np.array([1, 1, -2]),
            }
    gp = np.atleast_2d(result["glide_plane"])[0]
    result["orientation"] = np.roll(
        tools.orthonormalize([result["dislocation_line"], gp]), -1, axis=0
    )
    return result


def get_shockley_partials(
    burgers_vector: np.ndarray = np.array([1, -1, 0]),
    glide_plane: np.ndarray = np.array([1, 1, 1]),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the Shockley partials for a dislocation in fcc materials.

    Args:
        burgers_vector ((3,)-np.ndarray): Burgers vector.
        glide_plane ((3,)-np.ndarray): Glide plane normal.

    Returns:
        (np.ndarray, np.ndarray): Shockley partials.
    """
    assert np.shape(burgers_vector) == np.shape(glide_plane) == (3,)
    if not np.isclose(np.dot(burgers_vector, glide_plane), 0):
        raise ValueError("Burgers vector and glide plane are not orthogonal.")
    b = np.asarray(burgers_vector) / 2
    cross = np.cross(b, glide_plane) / np.linalg.norm(glide_plane) / np.sqrt(3)
    return b + cross, b - cross
