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

def teardown_module(module):
    cmd = ["rm -rf build"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

# Output
# simple program: 113.000000

def test_1():
    # --- compile code ---
    cmd = ['mkdir -p build && cd build && cmake ..']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ['cd build && ' + TRACING_TOLL_PATH + ' make -j']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    ## --- run code ---
    cmd = ["./build/src/main_program_fpc"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    res = cmdOutput.decode('utf-8').split()

    assert res[0] == 'simple'
    assert res[1] == 'program:'
    assert res[2] == '113.000000'

