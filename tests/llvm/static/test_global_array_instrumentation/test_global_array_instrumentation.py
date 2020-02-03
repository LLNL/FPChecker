#!/usr/bin/env python

import subprocess
import os

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean -f Makefile.0"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)


def check(cmdOutput):
    func_found = False
    global_array_created = False
    read_array_func_found = False
    write_array_func_found = False
    for l in cmdOutput:
        if "#FPCHECKER: Global errors array created" in l:
            global_array_created = True

        if "#FPCHECKER: Instrumenting function:" in l and "computed" in l:
            func_found = True

        if "#FPCHECKER: found function: _FPC_READ_GLOBAL_ERRORS_ARRAY_" in l:
            read_array_func_found = True

        if "#FPCHECKER: found function: _FPC_WRITE_GLOBAL_ERRORS_ARRAY" in l:
            write_array_func_found = True


    if func_found == True and global_array_created == True and read_array_func_found == True and write_array_func_found == True:
        passed = True
    else:
        passed = False

    return passed


def test_1():
    # -----------------------
    cmd = ["make -f Makefile.0"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    op0_passed = check(cmdOutput.decode('utf-8').split("\n"))
    # ------------------------

    # -----------------------
    cmd = ["make -f Makefile.1"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    op1_passed = check(cmdOutput.decode('utf-8').split("\n"))
    # ------------------------

    # -----------------------
    cmd = ["make -f Makefile.2"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    op2_passed = check(cmdOutput.decode('utf-8').split("\n"))
    # ------------------------

    # -----------------------
    cmd = ["make -f Makefile.3"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    op3_passed = check(cmdOutput.decode('utf-8').split("\n"))
    # ------------------------

    assert op0_passed == True
    assert op1_passed == True
    assert op2_passed == True 
    assert op3_passed == True
