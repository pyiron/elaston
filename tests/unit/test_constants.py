import unittest
from elaston.elastic_constants import ElasticConstants

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
    }
}


class TestConstants(unittest.TestCase):
    def test_is_cubic(self):
        ec = ElasticConstants(**data["Fe"])
        self.assertTrue(ec.is_cubic())
        ec = ElasticConstants(**data["W"])
        self.assertTrue(ec.is_cubic())
        ec = ElasticConstants(C_11=211.0, C_12=145.0, C_13=140, C_44=82.0)
        self.assertFalse(ec.is_cubic())

    def test_zener_ratio(self):
        ec = ElasticConstants(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertAlmostEqual(ec.get_zener_ratio(), 2 * 82 / (211 - 145))
        ec = ElasticConstants(C_11=211.0, C_12=145.0)
        self.assertAlmostEqual(ec.get_zener_ratio(), 1)
        ec = ElasticConstants(C_11=211.0, C_44=82.0)
        self.assertTrue(ec.is_isotropic())
        ec = ElasticConstants(C_11=211.0, C_12=145.0, C_13=140, C_44=82.0)
        self.assertRaises(ValueError, ec.get_zener_ratio)

    def test_voigt_average(self):
        ec = ElasticConstants(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertFalse(ec.is_isotropic())
        ec_ave = ec.get_voigt_average()
        self.assertTrue(ec_ave.is_isotropic())

if __name__ == "__main__":
    unittest.main()
