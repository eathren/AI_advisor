import os
import sys
import unittest
from util import read, write, file_exists


class TestAdvisor(unittest.TestCase):
    def test_files(self):
        data = {"1": 1, "2": 2}
        path = 'test.json'
        write(path, data)
        self.assertTrue(file_exists(path))
        read_data = read(path)
        self.assertEqual(data, read_data)
        os.remove(path)
        self.assertFalse(file_exists(path))

    def test_alpaca_reponse(self):
        pass

    def test_alpha_advantage_response(self):
        pass


if __name__ == '__main__':
    unittest.main()
