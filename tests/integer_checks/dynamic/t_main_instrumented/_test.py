
import subprocess
import unittest
import os

class TestFPChecker(unittest.TestCase):
    
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(THIS_DIR)
    
    def test_1(self):
        # --- compile code ---
        cmd = ["make -f Makefile.0 clean && make -f Makefile.0"]
        try:
            cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print e.output
            exit()
    
        # --- run code ---
        cmd = ["./main"]
        try:
            cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print e.output
            exit()

        n = 0
        output = cmdOutput.split("\n")
        for i in range(len(output)):
            if "========================================" in output[i]:
                if "FPChecker (v" in output[i+1]:
                    n = n + 1
        self.assertEqual(n, 1, "Only 1 banner at main() found")

if __name__ == '__main__':
    unittest.main()
