
import subprocess
import unittest
import os
import glob

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
        
                
        jsonFiles = glob.glob("*.json")
        self.assertEqual(len(jsonFiles), 1, "Only 1 json file generated")

        correctName = False
        if jsonFiles[0].split("_")[0] == 'fpc':
            correctName = True
        self.assertTrue(correctName, "Correct Json file name generated")

    def test_2(self):
        # --- run code ---
        cmd = ["./main"]
        try:
            cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print e.output
            exit()

        jsonFiles = glob.glob("*.json")
        fileName = jsonFiles[0]
        fd = open(fileName, 'r')
        lines = fd.readlines()
        fd.close()
        self.assertGreater(len(lines), 8, "Json file should not be empty")

if __name__ == '__main__':
    unittest.main()
