#!/usr/bin/env python

import test_config
import subprocess
import os
import sys

def main():
   
    print "* Static Tests *"
 
    ###########################################################################
    t = "Test: find instrumentation functions"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_find_inst_functions/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: num. fp operations"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_number_fp_operations/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: a device function is found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_device_func_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: a global function is found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_global_func_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: main() is found"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_main_is_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: global array instrumentation"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_global_array_instrumentation/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

    ###########################################################################
    t = "Test: correct func are found and instrumented"
    testTarget = test_config.textWidth.format(t)
    sys.stdout.write(testTarget)
    os.chdir(test_config.path + "/test_correct_inst_functions_found/")
    cmd = ["./test.py"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    sys.stdout.write(cmdOutput)
    os.chdir("../")
    ###########################################################################

main()
