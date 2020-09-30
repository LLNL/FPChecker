#!/usr/bin/env python

import subprocess
import os

TRACING_TOLL_PATH='../../../../../tracing_tool/fpchecker.py'

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["rm -rf build"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

# returns: tuple (error, op, file, line)
#
#+--------------------------- FPChecker Error Report ---------------------------+
# Error         : NaN                                                            
# Operation     : NONE                                                           
# File          : dot_product_copy.cu                                            
# Line          : 8                                                              
# Block (1,0,0), Thread (0,0,0)
#+------------------------------------------------------------------------------+
#
def getFPCReport(lines):
    ret = ("", "", "", "")
    for i in range(len(lines)):
        l = lines[i]
        if "FPChecker" in l and "Report" in l and "+" in l:
            err = lines[i+1].split()[2]
            op = lines[i+2].split()[2]
            f = lines[i+3].split()[2]
            line = lines[i+4].split()[2]
            ret = (err, op, f, line)
            break

    return ret

def run(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    return cmdOutput

def test_1():
    # --- compile code ---
    cmd = ['mkdir -p build && cd build && cmake -GNinja ..']
    run(cmd)

    cmd = ['cd build && ' + TRACING_TOLL_PATH + ' ninja']
    run(cmd)

    ## --- run code ---
    cmd = ["./build/src/main_fpc"]
    res = run(cmd)
    rep = getFPCReport(res.decode('utf-8').split('\n'))

    assert rep[0] == 'NaN' and rep[3] == '8'


