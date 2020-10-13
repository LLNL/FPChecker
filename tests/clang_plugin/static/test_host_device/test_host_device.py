#!/usr/bin/env python

import subprocess
import os

source = "compute.cu"
#  y += _FPC_CHECK_((x+1.25)*10.0, 8, "compute.cu");
#  *x = _FPC_CHECK_(*x / 1.2, 15, "compute.cu");
#  *x = _FPC_CHECK_((*x - 128.0) / (*x), 21, "compute.cu");
#  *x = _FPC_CHECK_((64+*y) * (*x), 27, "compute.cu");
expression1 = "(x+1.25)*10.0"
expression2 = "*x / 1.2"
expression3 = "(*x - 128.0) / (*x)"
expression4 = "(64+*y) * (*x)"

#expression1 = "(x + 1.25) * 10."
#expression2 = "*x / 1."
#expression3 = "(*x - 128.) / (*x)"
#expression4 = "(64 + *y) * (*x)"

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make -f Makefile.0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
    cmd = ["make -f Makefile.0"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    
    required_found1 = False
    required_found2 = False
    required_found3 = False
    required_found4 = False
    
    fd = open(source, "r")
    for l in fd:
        if "_FPC_CHECK_(" in l:
            if expression1 in l:
                loc1 = l.find("_FPC_CHECK_(")
                loc2 = l.find(expression1)
                loc3 = l.find(");")
                assert loc1 < loc2
                assert loc2 < loc3
                required_found1 = True
                
            if expression2 in l:
                loc1 = l.find("_FPC_CHECK_(")
                loc2 = l.find(expression2)
                loc3 = l.find(");")
                assert loc1 < loc2
                assert loc2 < loc3
                required_found2 = True
                
            if expression3 in l:
                loc1 = l.find("_FPC_CHECK_(")
                loc2 = l.find(expression3)
                loc3 = l.find(");")
                assert loc1 < loc2
                assert loc2 < loc3
                required_found3 = True
                
            if expression4 in l:
                loc1 = l.find("_FPC_CHECK_(")
                loc2 = l.find(expression4)
                loc3 = l.find(");")
                assert loc1 < loc2
                assert loc2 < loc3
                required_found4 = True
                
    fd.close()
    
    assert required_found1
    assert required_found2
    assert required_found3
    assert required_found4
