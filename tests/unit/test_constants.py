import unittest
from elaston.elastic_constants import ElasticConstants
import numpy as np

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
}


class TestConstants(unittest.TestCase):
    def test_consistency(self):
        E = 211
        nu = 0.29
        G = E / (2 * (1 + nu))
        ec = ElasticConstants(youngs_modulus=E, poissons_ratio=nu)
        self.assertTrue(
            np.allclose(
                ec.elastic_tensor,
                ElasticConstants(youngs_modulus=E, shear_modulus=G).elastic_tensor
            )
        )
        self.assertTrue(
            np.allclose(
                ec.elastic_tensor,
                ElasticConstants(poissons_ratio=nu, shear_modulus=G).elastic_tensor
            )
        )
        C_11 = 211.0
        C_12 = 145.0
        C_44 = (C_11 - C_12) / 2
        ec = ElasticConstants(C_11=C_11, C_12=C_12)
        self.assertTrue(
            np.allclose(
                ec.elastic_tensor,
                ElasticConstants(C_11=C_11, C_44=C_44).elastic_tensor
            )
        )
        self.assertTrue(
            np.allclose(
                ec.elastic_tensor,
                ElasticConstants(C_12=C_12, C_44=C_44).elastic_tensor
            )
        )
        self.assertRaises(
            ValueError, ElasticConstants, youngs_modulus=E, C_12=C_12, C_11=C_11
        )
        self.assertRaises(ValueError, ElasticConstants)
        self.assertRaises(ValueError, ElasticConstants, C_11=C_11)

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

    def test_unique_constants(self):
        ec = ElasticConstants(C_11=211.0, C_12=145.0, C_44=82.0)
        self.assertEqual(
            ec.get_unique_elastic_constants(),
            {"C_11": 211.0, "C_12": 145.0, "C_44": 82.0}
        )


if __name__ == "__main__":
    unittest.main()
