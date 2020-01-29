#!/usr/bin/env python

import subprocess
import os

source = "src/src/compute.cu"
expression = "x / (x + 1.3)"

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make -f Makefile.0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
    cmd = ["make -f Makefile.0"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    required_found = False
    fd = open(source, "r")
    for l in fd:
        if "_FPC_CHECK_(" in l:
            loc1 = l.find("_FPC_CHECK_(")
            " ".join(expression.split())
            loc2 = l.find(expression)
            loc3 = l.find(");")
            assert loc1 < loc2
            assert loc2 < loc3
            required_found = True
            break
    fd.close()
    
    assert required_found
