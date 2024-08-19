import numpy as np
import unittest
from elaston import tools
from pint import UnitRegistry


@tools.unit_wrapper("b.u / b.u * C.u", b="angstrom", x="angstrom", C="GPa")
def get_stress(b, x, C):
    return b / x * C


class TestTools(unittest.TestCase):
    def test_rotation(self):
        orientation = [[1, 1, 1], [-1, 1, 0], [1, -2, 1]]
        burgers_vector = np.array([1, 1, 1]) / np.sqrt(3)
        self.assertTrue(np.allclose(
            tools.crystal_to_box(burgers_vector, orientation),
            np.array([1, 0, 0])
        ))
        self.assertTrue(np.allclose(
            tools.box_to_crystal(
                tools.crystal_to_box(burgers_vector, orientation), orientation
            ),
            burgers_vector
        ))

    def test_orthonormalize(self):
        with self.assertRaises(ValueError):
            tools.orthonormalize([[1, 1, 1], [1, -1, 0], [1, -2, 1]])

if __name__ == "__main__":
    unittest.main()
