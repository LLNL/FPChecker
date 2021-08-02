#!/usr/bin/env python

import subprocess
import os
import sys

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["rm -rf build .fpc_logs"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

# Output
# simple program: 113.000000

def test_1():
    # --- compile code ---
    cmd = ['mkdir -p build && cd build && CXX=clang++ CC=clang cmake ..']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ['cd build && fpchecker make -j']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    ## --- run code ---
    cmd = ["./build/src/main_program"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    res = cmdOutput.decode('utf-8').split()

    assert res[0] == '#FPCHECKER:'
    assert res[1] == 'Initializing...'
    assert res[2] == 'simple'
    assert res[3] == 'program:'
    assert res[4] == '113.000000'

