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

    def test_pint_alone(self):
        ureg = UnitRegistry()
        self.assertEqual(
            (1 * ureg.meter / 1 * ureg.millisecond).magnitude,
            1/1e-3,
            msg="Does pint itself work on python 3.10?"
        )


if __name__ == "__main__":
    unittest.main()
