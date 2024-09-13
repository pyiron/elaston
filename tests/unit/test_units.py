import numpy as np
import unittest
from elaston.units import units, optional_units, u
from pint import UnitRegistry


@units
def get_speed_optional_arg(
    distance: u(float)["meter"],
    time: u(float)["second"],
    duration: u(float | None)["second"] = None,
) -> u(float)["meter/second"]:
    if duration is not None:
        return distance / duration
    return distance / time


@units
def get_speed_multiple_dispatch(
    distance: u(float)["meter"], time: u(float)["second"]
) -> u(float)["meter/second"]:
    return distance / time


@units()
def get_speed_multiple_types(
    distance: u(float | int)["meter"], time: u(float | int)["second"]
) -> u(float | int)["meter/second"]:
    return distance / time


@units(inputs={"b": "angstrom", "x": "angstrom", "C": "GPa"}, outputs="GPa")
def get_stress_absolute(b, x, C):
    return np.round(b / x * C, decimals=8)


@units(outputs=lambda b, x, C: b.u / x.u * C.u)
def get_stress_relative(b, x, C):
    return np.round(b / x * C, decimals=8)


@units(outputs=(lambda a, b: a.u, lambda a, b: a.u * b.u))
def get_multiple_outputs(a, b):
    return a + b, a * b


@units()
def no_units(a):
    return a


@units(
    outputs=lambda distance, time, duration: distance.u / optional_units(time, duration)
)
def get_velocity(distance, time=None, duration=None):
    if time is not None:
        return distance / time
    if duration is not None:
        return distance / duration


class TestTools(unittest.TestCase):
    def test_units(self):
        self.assertEqual(get_stress_absolute(1, 1, 1), 1)
        self.assertEqual(get_stress_relative(1, 1, 1), 1)
        ureg = UnitRegistry()
        self.assertEqual(
            get_stress_absolute(1 * ureg.angstrom, 1 * ureg.angstrom, 1 * ureg.GPa),
            1 * ureg.GPa,
        )
        self.assertEqual(
            get_stress_absolute(1 * ureg.nanometer, 1 * ureg.angstrom, 1 * ureg.GPa),
            10 * ureg.GPa,
        )
        self.assertEqual(
            get_stress_relative(1 * ureg.angstrom, 1 * ureg.angstrom, 1 * ureg.GPa),
            1 * ureg.GPa,
        )
        self.assertEqual(
            get_stress_relative(1 * ureg.nanometer, 1 * ureg.angstrom, 1 * ureg.GPa),
            1 * ureg.nanometer / ureg.angstrom * ureg.GPa,
        )
        with self.assertWarns(SyntaxWarning):
            get_stress_relative(1 * ureg.nanometer, 1 * ureg.angstrom, 1)
        self.assertEqual(
            get_multiple_outputs(1 * ureg.angstrom, 1 * ureg.angstrom),
            (2 * ureg.angstrom, 1 * ureg.angstrom**2),
        )
        with self.assertRaises(ValueError):
            # No relative output units and absolute input units at the same time
            units(outputs=lambda x: x.u, inputs={"x": "GPa"})
        self.assertEqual(no_units(1), 1)
        self.assertEqual(no_units(1 * ureg.angstrom), 1)

    def test_optional_units(self):
        ureg = UnitRegistry()
        self.assertEqual(
            get_velocity(distance=1 * ureg.angstrom, time=1 * ureg.second),
            1 * ureg.angstrom / ureg.second,
        )
        self.assertEqual(
            get_velocity(distance=1 * ureg.angstrom, duration=1 * ureg.second),
            1 * ureg.angstrom / ureg.second,
        )

    def test_type_hinting(self):
        self.assertEqual(get_speed_multiple_types(1, 1), 1)
        ureg = UnitRegistry()
        self.assertAlmostEqual(
            get_speed_multiple_types(1 * ureg.meter, 1 * ureg.millisecond).magnitude,
            1e3,
        )

    def test_multiple_dispatch(self):
        ureg = UnitRegistry()
        self.assertAlmostEqual(
            get_speed_multiple_dispatch(1 * ureg.meter, 1 * ureg.second).magnitude, 1
        )
        self.assertAlmostEqual(
            get_speed_multiple_dispatch(1 * ureg.meter, 1 * ureg.millisecond).magnitude,
            1e3,
        )

    def test_optional_arg(self):
        ureg = UnitRegistry()
        self.assertAlmostEqual(
            get_speed_optional_arg(1 * ureg.meter, 1 * ureg.second).magnitude,
            1/1,
            msg="Optional None kwarg should be ignored."
        )
        self.assertAlmostEqual(
            get_speed_optional_arg(
                1 * ureg.meter, 1 * ureg.second, 2 * ureg.second
            ).magnitude,
            1/2,
            msg="Optional kwarg should take precedence."
        )
        self.assertAlmostEqual(
            get_speed_optional_arg(1 * ureg.meter, 1 * ureg.millisecond).magnitude,
            1/1e-3,
            msg="Alternative units on standard arg should compute."
        )
        self.assertAlmostEqual(
            get_speed_optional_arg(
                1 * ureg.meter, 1 * ureg.millisecond, 2 * ureg.millisecond
            ).magnitude,
            1/2e-3,
            msg="Alternative units from preferred optional kwarg should compute."
        )


if __name__ == "__main__":
    unittest.main()
