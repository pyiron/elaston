import unittest

import numpy as np

from elaston import orientation


class TestOrientation(unittest.TestCase):
    def test_get_dislocation_orientation(self):
        for t in ["bcc", "fcc"]:
            orient = orientation.get_dislocation_orientation(
                dislocation_type="screw", crystal=t
            )
            self.assertEqual(
                orient["dislocation_line"].tolist(), orient["burgers_vector"].tolist()
            )
            for d in ["screw", "edge"]:
                orient = orientation.get_dislocation_orientation(
                    dislocation_type=d, crystal=t
                )
                for gp in np.atleast_2d(orient["glide_plane"]):
                    self.assertAlmostEqual(np.dot(orient["dislocation_line"], gp), 0)

    def test_get_shockley_partials(self):
        b_1, b_2 = orientation.get_shockley_partials(
            burgers_vector=[-0.5, 0.5, 0], glide_plane=[1, 1, 1]
        )
        self.assertAlmostEqual(np.linalg.norm(6 * b_1 - [-1, 2, -1]), 0)
        self.assertAlmostEqual(np.linalg.norm(6 * b_2 - [-2, 1, 1]), 0)
        for b in [[1, 1, 0], [1, 0, -1], [0, 1, 1]]:
            for sign in [1, -1]:
                b_1, b_2 = orientation.get_shockley_partials(
                    burgers_vector=sign * np.asarray(b), glide_plane=[1, -1, 1]
                )
                self.assertTrue(np.allclose(b_1 + b_2, sign * np.asarray(b)))
                self.assertAlmostEqual(np.dot(b_1, [1, -1, 1]), 0)
        self.assertRaises(
            ValueError, orientation.get_shockley_partials, [1, 1, 0], [1, 1, 1]
        )


if __name__ == "__main__":
    unittest.main()
