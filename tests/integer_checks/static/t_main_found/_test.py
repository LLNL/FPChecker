
import subprocess
import unittest
import os

class TestFPChecker(unittest.TestCase):
    
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(THIS_DIR)
    
    def test_1(self):
        cmd = ["make -f Makefile.0"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        n = 0
        for l in cmdOutput.split("\n"):
            if "#FPCHECKER: main() found" in l:
                n = n + 1
        self.assertEqual(n, 1, "Only 1 main() func is found")

if __name__ == '__main__':
    unittest.main()
