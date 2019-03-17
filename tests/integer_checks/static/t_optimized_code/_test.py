
import subprocess
import unittest
import os

class TestFPChecker(unittest.TestCase):
    
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(THIS_DIR)
    
    def test_1(self):
        n = 0
        cmd = ["make -f Makefile.0"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        out = cmdOutput.split("\n")
        for i in range(len(out)):
            if "#FPCHECKER: main() found" in out[i]:
                n = n + 1
        
        cmd = ["make -f Makefile.1"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        out = cmdOutput.split("\n")
        for i in range(len(out)):
            if "#FPCHECKER: main() found" in out[i]:
                n = n + 1

        cmd = ["make -f Makefile.2"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        out = cmdOutput.split("\n")
        for i in range(len(out)):
            if "#FPCHECKER: main() found" in out[i]:
                n = n + 1

        cmd = ["make -f Makefile.3"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        out = cmdOutput.split("\n")
        for i in range(len(out)):
            if "#FPCHECKER: main() found" in out[i]:
                n = n + 1
        
        self.assertEqual(n, 4, "Should produce 4 programs without errors")

if __name__ == '__main__':
    unittest.main()
