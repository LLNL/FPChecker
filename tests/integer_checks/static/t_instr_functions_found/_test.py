
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
        f = ["_FPC_HT_CREATE_", "_FPC_HT_HASH_", "_FPC_HT_NEWPAIR_", "_FPC_ITEMS_EQUAL_", "_FPC_HT_SET_", "_FPC_PRINT_HASH_TABLE_", "_FPC_INIT_HASH_TABLE_", "_FPC_PRINT_LOCATIONS_", "_FPC_CHECK_OVERFLOW_", "_FPC_FP32_CHECK_ADD_", "_FPC_FP32_CHECK_SUB_", "_FPC_FP32_CHECK_MUL_", "_FPC_FP32_CHECK_DIV_"]
        n = 0
        for l in cmdOutput.split("\n"):
            if "#FPCHECKER: Found " in l:
                v = l.split()[2]
                if v in f:
                    n = n + 1
        self.assertEqual(n, 13, "Should find 14 functions")

    def test_2(self):
        cmd = ["make -f Makefile.1"]
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        f = ["_FPC_HT_CREATE_", "_FPC_HT_HASH_", "_FPC_HT_NEWPAIR_", "_FPC_ITEMS_EQUAL_", "_FPC_HT_SET_", "_FPC_PRINT_HASH_TABLE_", "_FPC_UNUSED_FUNC_", "_FPC_INIT_HASH_TABLE_", "_FPC_PRINT_LOCATIONS_", "_FPC_CHECK_OVERFLOW_", "_FPC_FP32_CHECK_ADD_", "_FPC_FP32_CHECK_SUB_", "_FPC_FP32_CHECK_MUL_", "_FPC_FP32_CHECK_DIV_"]
        n = 0
        for l in cmdOutput.split("\n"):
            if "#FPCHECKER: Found " in l:
                v = l.split()[2]
                if v in f:
                    n = n + 1
        self.assertEqual(n, 0, "Should find 0 functions")

if __name__ == '__main__':
    unittest.main()
