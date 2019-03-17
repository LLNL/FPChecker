
#import subprocess
from subprocess import Popen, PIPE
import unittest
import os
import glob
import json


class TestFPChecker(unittest.TestCase):
    
    def setUp(self):
        THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        os.chdir(THIS_DIR)
    
    def test_1(self):
        #p = Popen(['make', '-f', 'Makefile.0', 'clean'], stdout=PIPE, stderr=PIPE)
        p = Popen(['make', '-f', 'Makefile.0'], stdout=PIPE, stderr=PIPE)
        output, error = p.communicate()
        
        correctOutput = False
        if "#FPCHECKER: Make sure program is compiled with -g" in error:
            correctOutput = True
        
        self.assertTrue(correctOutput, "Should indicate to compile with -g")


if __name__ == '__main__':
    unittest.main()
