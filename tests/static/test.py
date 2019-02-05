#!/usr/bin/env python

import subprocess
import os
import sys

def main():
    
    ###########################################################################
    sys.stdout.write("Test: find instrumentation functions ")
    os.chdir("./test_find_inst_functions/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    sys.stdout.write("Test: num. fp operations ")
    os.chdir("./test_number_fp_operations/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    sys.stdout.write("Test: a device function is found ")
    os.chdir("./test_device_func_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    sys.stdout.write("Test: a global function is found ")
    os.chdir("./test_global_func_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

main()
