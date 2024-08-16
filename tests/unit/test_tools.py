import numpy as np
import unittest
from elaston import tools


class TestTools(unittest.TestCase):
    def test_rotation(self):
        orientation = [[1, 1, 1], [1, -1, 0], [1, -2, 1]]
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

if __name__ == "__main__":
    unittest.main()
