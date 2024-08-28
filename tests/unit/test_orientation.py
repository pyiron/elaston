import numpy as np
import unittest
from elaston import orientation


class TestOrientation(unittest.TestCase):
    def test_get_dislocation_orientation(self):
        for t in ["bcc", "fcc"]:
            orient = orientation.get_dislocation_orientation(
                dislocation_type="screw", crystal=t
            )
            self.assertEqual(orient["dislocation_line"], orient["burgers_vector"])
            for d in ["screw", "edge"]:
                orient = orientation.get_dislocation_orientation(
                    dislocation_type=d, crystal=t
                )
                for gp in np.atleast_2d(orient["glide_plane"]):
                    self.assertAlmostEqual(np.dot(orient["dislocation_line"], gp), 0)


if __name__ == "__main__":
    unittest.main()
