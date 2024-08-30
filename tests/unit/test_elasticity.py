import numpy as np
import unittest
from elaston.linear_elasticity import LinearElasticity
from elaston import tools


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


class TestElasticity(unittest.TestCase):
    def test_frame(self):
        medium = LinearElasticity(np.random.random((6, 6)))
        self.assertIsNone(medium.orientation)
        medium.orientation = 0.1 * np.random.randn(3, 3) + np.eye(3)
        self.assertAlmostEqual(np.linalg.det(medium.orientation), 1)
        self.assertRaises(ValueError, setattr, medium, "orientation", -np.eye(3))

    def test_orientation(self):
        elastic_tensor = create_random_C()
        epsilon = np.random.random((3, 3))
        epsilon += epsilon.T
        sigma = np.einsum("ijkl,kl->ij", elastic_tensor, epsilon)
        medium = LinearElasticity(
            elastic_tensor, orientation=np.array([[1, 1, 1], [1, 0, -1]])
        )
        orientation = medium.orientation
        self.assertAlmostEqual(np.linalg.det(orientation), 1)
        medium.orientation = np.array([[1, 1, 1], [1, 0, -1]])
        self.assertTrue(np.allclose(orientation, medium.orientation))
        sigma = np.einsum("iI,jJ,IJ->ij", medium.orientation, medium.orientation, sigma)
        sigma_calc = np.einsum(
            "ijkl,kK,lL,KL->ij",
            medium.get_elastic_tensor(),
            medium.orientation,
            medium.orientation,
            epsilon,
        )
        self.assertTrue(np.allclose(sigma - sigma_calc, 0))

    def test_elastic_constants(self):
        medium = LinearElasticity(np.eye(6))
        self.assertRaises(ValueError, medium.get_elastic_moduli)
        medium = LinearElasticity(C_11=211.0, C_12=130.0)
        param = medium.get_elastic_moduli()
        for key in [
            "bulk_modulus",
            "shear_modulus",
            "youngs_modulus",
            "poissons_ratio",
        ]:
            self.assertIn(key, param)

    def test_isotropic(self):
        medium = LinearElasticity(create_random_C(isotropic=True))
        self.assertTrue(medium.is_isotropic())
        medium = LinearElasticity(create_random_C(isotropic=False))
        self.assertFalse(medium.is_isotropic())
        medium = medium.get_voigt_average()
        self.assertTrue(medium.is_isotropic())
        medium = LinearElasticity(create_random_C(isotropic=False))
        medium = medium.get_reuss_average()
        self.assertTrue(medium.is_isotropic())

    def test_compliance_tensor(self):
        elastic_tensor = create_random_C()
        medium = LinearElasticity(elastic_tensor)
        compliance = medium.get_compliance_tensor(voigt=True)
        self.assertTrue(
            np.allclose(
                np.linalg.inv(medium.get_elastic_tensor(voigt=True)), compliance
            )
        )
        E = 0.5 * np.einsum("ik,jl->ijkl", *2 * [np.eye(3)])
        E += 0.5 * np.einsum("il,jk->ijkl", *2 * [np.eye(3)])
        self.assertTrue(
            np.allclose(
                np.einsum(
                    "ijkl,klmn->ijmn",
                    medium.get_compliance_tensor(voigt=False),
                    medium.get_elastic_tensor(voigt=False),
                ),
                E,
            )
        )

    def test_dislocation_energy(self):
        elastic_tensor = create_random_C()
        medium = LinearElasticity(elastic_tensor)
        r_max = 1e6 * np.random.random() + 10
        r_min_one = 10 * np.random.random()
        r_min_two = 10 * np.random.random()
        E_one = medium.get_dislocation_energy([0, 0, 1], r_min_one, r_max)
        E_two = medium.get_dislocation_energy([0, 0, 1], r_min_two, r_max)
        self.assertGreater(E_one, 0)
        self.assertGreater(E_two, 0)
        self.assertAlmostEqual(
            E_one / np.log(r_max / r_min_one), E_two / np.log(r_max / r_min_two)
        )

    def test_dislocation_force(self):
        elastic_tensor = create_random_C()
        medium = LinearElasticity(elastic_tensor)
        medium.orientation = [[1, -2, 1], [1, 1, 1], [-1, 0, 1]]
        lattice_constant = 3.52
        partial_one = np.array([-0.5, 0, np.sqrt(3) / 2]) * lattice_constant
        partial_two = np.array([0.5, 0, np.sqrt(3) / 2]) * lattice_constant
        stress = medium.get_dislocation_stress([0, 10, 0], partial_one)
        force = medium.get_dislocation_force(stress, [0, 1, 0], partial_two)
        self.assertAlmostEqual(force[1], 0)
        self.assertAlmostEqual(force[2], 0)

    def test_dislocation_stress(self):
        medium = LinearElasticity(C_11=211.0, C_12=130.0, C_44=82.0)
        dx = 1e-7
        x = np.array([[0, 0, 0], [dx, 0, 0], [0, dx, 0], [0, 0, dx]]) + np.ones(3)
        y = medium.get_dislocation_displacement(x, np.ones(3))
        eps = (y[1:] - y[0]) / dx
        eps = 0.5 * (eps + eps.T)
        self.assertTrue(
            np.allclose(eps, medium.get_dislocation_strain(x, np.ones(3)).mean(axis=0))
        )
        x = np.random.randn(3) + [10, 1, 1]
        strain = medium.get_dislocation_strain(x, np.ones(3))
        stress = np.einsum("ijkl,kl->ij", medium.get_elastic_tensor(), strain)
        self.assertTrue(
            np.allclose(stress, medium.get_dislocation_stress(x, np.ones(3)))
        )
        x = np.random.randn(10, 3)
        self.assertGreater(
            medium.get_dislocation_energy_density(x, np.ones(3)).min(), 0
        )

    def test_elastic_tensor_input(self):
        C = create_random_C()
        medium = LinearElasticity(
            C_11=C[0, 0, 0, 0],
            C_12=C[0, 0, 1, 1],
            C_44=C[0, 1, 0, 1],
            orientation=[[1, 1, 1], [1, 0, -1]],
        )
        self.assertEqual(
            medium.get_elastic_tensor(voigt=False, rotate=True).shape, (3, 3, 3, 3)
        )
        self.assertEqual(
            medium.get_elastic_tensor(voigt=True, rotate=True).shape, (6, 6)
        )
        self.assertEqual(
            medium.get_elastic_tensor(voigt=False, rotate=False).shape, (3, 3, 3, 3)
        )
        self.assertEqual(
            medium.get_elastic_tensor(voigt=True, rotate=False).shape, (6, 6)
        )
        self.assertTrue(
            np.allclose(C, medium.get_elastic_tensor(voigt=False, rotate=False))
        )
        self.assertFalse(
            np.allclose(C, medium.get_elastic_tensor(voigt=False, rotate=True))
        )
        self.assertRaises(ValueError, LinearElasticity, np.random.random((3, 3)))

    def test_point_defect(self):
        for d in [
            {"C_11": 211.0, "C_12": 130.0, "C_44": 82.0},
            {"C_11": 211.0, "C_12": 130.0},
        ]:
            medium = LinearElasticity(**d)
            self.assertEqual(
                medium.get_point_defect_displacement(np.ones(3), np.eye(3)).shape, (3,)
            )
            dx = 1e-7
            x = np.array([[0, 0, 0], [dx, 0, 0], [0, dx, 0], [0, 0, dx]]) + np.ones(3)
            y = medium.get_point_defect_displacement(x, np.eye(3))
            eps = (y[1:] - y[0]) / dx
            eps = 0.5 * (eps + eps.T)
            self.assertTrue(
                np.allclose(
                    eps, medium.get_point_defect_strain(x, np.eye(3)).mean(axis=0)
                )
            )
            x = np.random.randn(3)
            strain = medium.get_point_defect_strain(x, np.eye(3))
            stress = np.einsum("ijkl,kl->ij", medium.get_elastic_tensor(), strain)
            self.assertTrue(
                np.allclose(stress, medium.get_point_defect_stress(x, np.eye(3)))
            )
            x = np.random.randn(10, 3)
            self.assertGreater(
                medium.get_point_defect_energy_density(x, np.eye(3)).min(), 0
            )


if __name__ == "__main__":
    unittest.main()
