#!/usr/bin/env python

import subprocess
import os
import sys
sys.path.append("..")
from .. import conftest

TRACING_TOLL_PATH = conftest.FPCHECKER_PATH

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

# returns: tuple (error, op, file, line)
#
#+--------------------------- FPChecker Error Report ---------------------------+
# Error         : NaN                                                            
# Operation     : NONE                                                           
# File          : dot_product_copy.cu                                            
# Line          : 7                                                              
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


def teardown_module(module):
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_exit(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        return cmdOutput
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

def test_1():
    # --- compile code ---
    cmd = [TRACING_TOLL_PATH + ' make']
    out = run_exit(cmd)
    # Failed commands: 2 
    out = ' '.join(out.decode('utf-8').split('\n')[-2:])
    print(out)
    assert 'Failed commands: 2' in out

    cmd = ["./main_fpc"]
    output = run_exit(cmd) 
    rep = getFPCReport(output.decode('utf-8').split("\n"))
    assert rep[0] == 'NaN' and rep[3] == '7'

