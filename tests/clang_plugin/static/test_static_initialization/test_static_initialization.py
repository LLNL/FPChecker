#!/usr/bin/env python

import subprocess
import os

source = "compute.cu"
expression1 = "energy + 10.5"
expression2 = "(x + (int)128)"

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
    
    numbwerOfTransformations = 0
    fd = open(source, "r")
    for l in fd:
        if "_FPC_CHECK_(" in l:
            numbwerOfTransformations = numbwerOfTransformations + 1
            
            if expression1 in l:
                loc1 = l.find("y +=")
                loc2 = l.find("_FPC_CHECK_(")
                loc3 = l.find(expression1)
                loc4 = l.find(");")
                assert loc1 < loc2
                assert loc2 < loc3
                assert loc3 < loc4
                required_found1 = True
                
            if expression2 in l:
                loc1 = l.find("_FPC_CHECK_(")
                loc2 = l.find(expression2)
                loc3 = l.find(");")
                assert loc1 < loc2
                assert loc2 < loc3
                required_found2 = True
    fd.close()
    
    assert required_found1
    assert required_found2
    assert numbwerOfTransformations == 2
