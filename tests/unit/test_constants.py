import unittest
from elaston import elastic_constants as ec
import numpy as np
from pint import UnitRegistry

ureg = UnitRegistry()

data = {
    "Fe": {
        "youngs_modulus": 211.0,
        "poissons_ratio": 0.29,
        "shear_modulus": 82.0,
    },
    "W": {
        "youngs_modulus": 411.0,
        "poissons_ratio": 0.28,
        "shear_modulus": 161.0,
    },
    "Si": {
        "youngs_modulus": 130.0,
        "poissons_ratio": 0.28,
        "shear_modulus": 51.0,
    },
    "Ni": {  # from materialsproject
        "C_11": 249.0,
        "C_12": 136.0,
        "C_44": 127.0,
    },
}


class TestConstants(unittest.TestCase):
    def test_consistency(self):
        E = 211
        nu = 0.29
        mu = E / (2 * (1 + nu))
        self.assertTrue(
            np.allclose(
                ec.get_elastic_tensor_from_moduli(E=E, nu=nu),
                ec.get_elastic_tensor_from_moduli(E=E, mu=mu),
            )
        )
        self.assertTrue(
            np.allclose(
                ec.get_elastic_tensor_from_moduli(E=E, nu=nu),
                ec.get_elastic_tensor_from_moduli(nu=nu, mu=mu),
            )
        )
        C_11 = 211.0
        C_12 = 145.0
        C_44 = (C_11 - C_12) / 2
        self.assertTrue(
            np.allclose(
                ec.get_elastic_tensor_from_tensor(C_11=C_11, C_12=C_12),
                ec.get_elastic_tensor_from_tensor(C_11=C_11, C_44=C_44),
            )
        )
        self.assertTrue(
            np.allclose(
                ec.get_elastic_tensor_from_tensor(C_11=C_11, C_12=C_12),
                ec.get_elastic_tensor_from_tensor(C_12=C_12, C_44=C_44),
            )
        )
        self.assertRaises(
            ValueError,
            ec.initialize_elastic_tensor,
            youngs_modulus=E,
            C_12=C_12,
            C_11=C_11,
        )
        self.assertRaises(ValueError, ec.get_elastic_tensor_from_moduli)
        self.assertRaises(ValueError, ec.get_elastic_tensor_from_tensor)
        self.assertRaises(ValueError, ec.initialize_elastic_tensor)
        self.assertRaises(ValueError, ec.initialize_elastic_tensor, C_11=C_11)
        self.assertRaises(
            ValueError, ec.initialize_elastic_tensor, C_tensor=np.arange(6)
        )
        C = ec.initialize_elastic_tensor(C_tensor=np.eye(6))
        self.assertEqual(C.tolist(), np.eye(6).tolist())
        C = ec.initialize_elastic_tensor(C_tensor=np.eye(9).reshape(3, 3, 3, 3))
        self.assertEqual(C.tolist(), np.eye(6).tolist())

    def test_is_cubic(self):
        C = ec.initialize_elastic_tensor(**data["Fe"])
        self.assertTrue(ec.is_cubic(C))
        C = ec.initialize_elastic_tensor(**data["W"])
        self.assertTrue(ec.is_cubic(C))
        C = ec.initialize_elastic_tensor(
            C_11=211.0, C_12=145.0, C_13=140, C_44=82.0
        )
        self.assertFalse(ec.is_cubic(C))

    def test_zener_ratio(self):
        C = ec.initialize_elastic_tensor(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertAlmostEqual(ec.get_zener_ratio(C), 2 * 82 / (211 - 145))
        C = ec.initialize_elastic_tensor(C_11=211.0, C_12=145.0)
        self.assertAlmostEqual(ec.get_zener_ratio(C), 1)
        C = ec.initialize_elastic_tensor(C_11=211.0, C_44=82.0)
        self.assertTrue(ec.is_isotropic(C))
        C = ec.initialize_elastic_tensor(
            C_11=211.0 * ureg.gigapascal, C_44=82.0 * ureg.gigapascal
        )
        self.assertTrue(ec.is_isotropic(C))
        C = ec.initialize_elastic_tensor(
            C_11=211.0, C_12=145.0, C_13=140, C_44=82.0
        )
        self.assertRaises(ValueError, ec.get_zener_ratio, C)

    def test_voigt_average(self):
        C = ec.initialize_elastic_tensor(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertFalse(ec.is_isotropic(C))
        C = ec.initialize_elastic_tensor(**ec.get_voigt_average(C))
        self.assertTrue(ec.is_isotropic(C))

    def test_reuss_average(self):
        C = ec.initialize_elastic_tensor(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertFalse(ec.is_isotropic(C))
        C = ec.initialize_elastic_tensor(**ec.get_reuss_average(C))
        self.assertTrue(ec.is_isotropic(C))

    def test_unique_constants(self):
        C = ec.initialize_elastic_tensor(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertEqual(
            ec.get_unique_elastic_constants(C),
            {"C_11": 211.0, "C_12": 145.0, "C_44": 82.0},
        )

    def test_elastic_moduli(self):
        C = ec.initialize_elastic_tensor(**data["Ni"])
        self.assertRaises(
            ValueError,
            ec.get_elastic_moduli,
            C,
            msg="Not isotropic and therefore no unique moduli",
        )
        C = ec.initialize_elastic_tensor(**ec.get_reuss_average(C))
        moduli = ec.get_elastic_moduli(C)
        self.assertLess(np.absolute(moduli["bulk_modulus"] - 174.0), 1)
        self.assertLess(np.absolute(moduli["shear_modulus"] - 85.0), 1)
        C = ec.initialize_elastic_tensor(**data["Ni"])
        C = ec.initialize_elastic_tensor(**ec.get_voigt_average(C))
        moduli = ec.get_elastic_moduli(C)
        self.assertLess(np.absolute(moduli["bulk_modulus"] - 174.0), 1)
        self.assertLess(np.absolute(moduli["shear_modulus"] - 99.0), 1)


if __name__ == "__main__":
    unittest.main()
