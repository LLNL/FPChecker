#!/usr/bin/env python

import test_config
import subprocess
import os
import sys

def main():

    print "* Dynamic Tests *"
    
    ###########################################################################
    t = "Test: fp32 underflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir("./test_fp32_underflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp64 underflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir("./test_fp64_underflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

main()
