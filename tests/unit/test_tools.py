import numpy as np
import unittest
from elaston import tools
from pint import UnitRegistry


@tools.units(
    inputs={"b": "angstrom", "x": "angstrom", "C": "GPa"}
)
def get_stress_absolute(b, x, C):
    return np.round(b / x * C, decimals=8)


@tools.units(outputs=lambda b, x, C: b.u / x.u * C.u)
def get_stress_relative(b, x, C):
    return np.round(b / x * C, decimals=8)


@tools.units(outputs=(lambda a, b: a.u, lambda a, b: a.u * b.u))
def get_multiple_outputs(a, b):
    return a + b, a * b

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
        self.assertEqual(get_stress_absolute(1, 1, 1), 1)
        self.assertEqual(get_stress_relative(1, 1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_stress_absolute(
                1 * ureg.angstrom, 1 * ureg.angstrom, 1 * ureg.GPa
            ),
            1
        )
        self.assertEqual(
            get_stress_absolute(
                1 * ureg.nanometer, 1 * ureg.angstrom, 1 * ureg.GPa
            ),
            10
        )
        self.assertEqual(
            get_stress_relative(
                1 * ureg.angstrom, 1 * ureg.angstrom, 1 * ureg.GPa
            ),
            1 * ureg.GPa
        )
        self.assertEqual(
            get_stress_relative(
                1 * ureg.nanometer, 1 * ureg.angstrom, 1 * ureg.GPa
            ),
            1 * ureg.nanometer / ureg.angstrom * ureg.GPa
        )
        with self.assertRaises(SyntaxError):
            get_stress_relative(1 * ureg.nanometer, 1 * ureg.angstrom, 1)
        self.assertEqual(
            get_multiple_outputs(1 * ureg.angstrom, 1 * ureg.angstrom),
            (2 * ureg.angstrom, 1 * ureg.angstrom ** 2)
        )


if __name__ == "__main__":
    unittest.main()
