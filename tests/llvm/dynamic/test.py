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
    os.chdir(test_config.path + "/test_fp32_underflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp64 underflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp64_underflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp32 NaN found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp32_nan_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp64 NaN found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp64_nan_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp32 overflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp32_overflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp64 overflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp64_overflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp32 almost underflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp32_almost_underflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: fp64 almost underflow found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_fp64_almost_underflow_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: debug info is correct"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_debug_info_is_correct/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: operations are correct"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_ops_are_correct/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: print at main"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_print_at_main/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: errors dont abort"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_errors_dont_abort/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: warnings dont abort"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_warnings_dont_abort/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: raja examples"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_raja_examples/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

main()
