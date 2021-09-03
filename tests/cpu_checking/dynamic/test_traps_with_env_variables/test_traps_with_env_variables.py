#!/usr/bin/env python

import subprocess
import os
import sys
from dynamic import report

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_command(cmd):
    ret = 0
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        ret = e.returncode
    return ret

def test_1():
    # --- compile code ---
    cmd = ["make"]
    run_command(cmd)

    #####  Tests ####

    ### Non failure cases
    cmd = ["FPC_TRAP_INFINITY_NEG=1 ./main"]
    assert run_command(cmd) == 0
    
    cmd = ["FPC_TRAP_INFINITY_NEG=1 FPC_TRAP_FILE=/path/file.cpp ./main"]
    assert run_command(cmd) == 0

    cmd = ["FPC_TRAP_INFINITY_NEG=1 FPC_TRAP_FILE=compute.cpp ./main"]
    assert run_command(cmd) == 0
    
    cmd = ["FPC_TRAP_CANCELLATION=1 FPC_TRAP_LINE=14 ./main"]
    assert run_command(cmd) == 0

    ### Failure Cases ###
    cmd = ["FPC_TRAP_NAN=1 ./main"]
    assert run_command(cmd) != 0

    cmd = ["FPC_TRAP_INFINITY_POS=1 ./main"]
    assert run_command(cmd) != 0

    cmd = ["FPC_TRAP_FILE=compute.cpp FPC_TRAP_NAN=1 ./main"]
    assert run_command(cmd) != 0

    cmd = ["FPC_TRAP_FILE=compute.cpp FPC_TRAP_INFINITY_POS=1 ./main"]
    assert run_command(cmd) != 0

    cmd = ["FPC_TRAP_CANCELLATION=1 FPC_TRAP_LINE=15 ./main"]
    assert run_command(cmd) != 0

