
import subprocess
import unittest
import os

class TestFPChecker(unittest.TestCase):
    
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(THIS_DIR)
    
    def test_1(self):
        cmd = ["make"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        n = 0
        out = cmdOutput.split("\n")
        v = 0
        for i in range(len(out)):
            if "#FPCHECKER: [ host function ] _Z4compii" in out[i]:
                v = int(out[i+2].split()[3])
        
        self.assertEqual(v, 5, "Should find 5 operations")

if __name__ == '__main__':
    unittest.main()
