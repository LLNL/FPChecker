
import subprocess
import unittest
import os
import glob
import json


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
        fileName = jsonFiles[0]

        with open(fileName) as f:
            data = json.load(f)
        self.assertGreater(len(data), 1, "Only 1 json file generated")

        n = 0
        for d in data:
            if d['over'] == 1:
                n = n + d['over']
        self.assertEqual(n, 1, "One overflow should be found")

    def test_2(self):
        jsonFiles = glob.glob("*.json")
        fileName = jsonFiles[0]
        
        with open(fileName) as f:
            data = json.load(f)

        n = 0
        correctResult = False
        for d in data:
            if d['over'] == 1:
                if d['over_res'] < -2147483648 or d['over_res'] > 2147483647:
                    correctResult = True
        self.assertTrue(correctResult, "Result of oveflow is correct")

if __name__ == '__main__':
    unittest.main()
