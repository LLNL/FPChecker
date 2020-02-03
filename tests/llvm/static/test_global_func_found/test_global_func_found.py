#!/usr/bin/env python

import subprocess
import os

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
    cmd = ["make"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

    passed = False
    for l in cmdOutput.decode('utf-8').split("\n"):
        if "#FPCHECKER: Instrumenting function:" in l and "computed" in l:
            passed = True

    assert passed == True

