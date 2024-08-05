import unittest
import elaston


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = elaston.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
