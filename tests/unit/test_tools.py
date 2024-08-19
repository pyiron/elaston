import numpy as np
import unittest
from elaston import tools
from pint import UnitRegistry


@tools.units(
    outputs="b / x * C", inputs={"b": "angstrom", "x": "angstrom", "C": "GPa"}
)
def get_stress(b, x, C):
    return np.round(b / x * C, decimals=8)


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

    def test_units(self):
        self.assertEqual(get_stress(1, 1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_stress(1 * ureg.angstrom, 1 * ureg.angstrom, 1 * ureg.GPa).magnitude,
            1
        )
        self.assertEqual(
            get_stress(1 * ureg.nanometer, 1 * ureg.angstrom, 1 * ureg.GPa).magnitude,
            10
        )
        with self.assertRaises(SyntaxError):
            get_stress(1 * ureg.nanometer, 1 * ureg.angstrom, 1)

if __name__ == "__main__":
    unittest.main()
