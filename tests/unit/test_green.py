import unittest

import numpy as np

from elaston import elastic_constants as ec
from elaston import tools
from elaston.green import Anisotropic, Isotropic


def create_random_C(isotropic=False):
    C11_range = np.array([0.7120697386322292, 1.5435656086034886])
    coeff_C12 = np.array([0.65797601, -0.0199679])
    coeff_C44 = np.array([0.72753844, -0.30418746])
    C = np.zeros((6, 6))
    C11 = C11_range[0] + np.random.random() * np.ptp(C11_range)
    C12 = np.polyval(coeff_C12, C11) + 0.2 * (np.random.random() - 0.5)
    C44 = np.polyval(coeff_C44, C11) + 0.2 * (np.random.random() - 0.5)
    C[:3, :3] = C12
    C[:3, :3] += (C11 - C12) * np.eye(3)
    if isotropic:
        C[3:, 3:] = np.eye(3) * (C[0, 0] - C[0, 1]) / 2
    else:
        C[3:, 3:] = C44 * np.eye(3)
    return tools.C_from_voigt(C)


class TestGreen(unittest.TestCase):
    def test_derivative(self):
        C = create_random_C()
        positions = np.tile(np.random.random(3), 2).reshape(-1, 3)
        dz = 1.0e-6
        index = np.random.randint(3)
        positions[1, index] += dz
        G_an = Anisotropic(C).get_greens_function(positions.mean(axis=0), derivative=1)[
            :, index, :
        ]
        G_num = np.diff(Anisotropic(C).get_greens_function(positions), axis=0) / dz
        self.assertTrue(np.isclose(G_num - G_an, 0).all())
        G_an = Anisotropic(C).get_greens_function(positions.mean(axis=0), derivative=2)[
            :, :, :, index
        ]
        G_num = (
            np.diff(Anisotropic(C).get_greens_function(positions, derivative=1), axis=0)
            / dz
        )
        self.assertTrue(np.isclose(G_num - G_an, 0).all())

    def test_comp_iso_aniso(self):
        shear_modulus = 52.5
        lame_parameter = 101.3
        poissons_ratio = 0.5 / (1 + shear_modulus / lame_parameter)
        C_11 = lame_parameter + 2 * shear_modulus
        C_12 = lame_parameter
        C_44 = shear_modulus
        iso = Isotropic(poissons_ratio, shear_modulus)
        aniso = Anisotropic(
            tools.C_from_voigt(
                ec.get_elastic_tensor_from_tensor(C_11=C_11, C_12=C_12, C_44=C_44)
            )
        )
        x = np.random.randn(100, 3) * 10
        for i in range(3):
            self.assertLess(
                np.ptp(
                    iso.get_greens_function(x, derivative=i)
                    - aniso.get_greens_function(x, derivative=i)
                ),
                1e-8,
                msg=f"Aniso- and isotropic Green's F's give different results for derivative={i}",
            )

    def test_memory(self):
        aniso = Anisotropic(create_random_C())
        positions = np.tile(np.random.random(3), 2).reshape(-1, 3)
        G_normal = aniso.get_greens_function(positions)
        G_unique = aniso.get_greens_function(positions, check_unique=True)
        self.assertTrue(np.all(np.isclose(G_normal, G_unique)))


if __name__ == "__main__":
    unittest.main()
