#!/usr/bin/env python

import subprocess
import os

# Instrumnted code
#__device__
#double compute(double x)
#{
#  double y = 0.0, z = 2.0;
#  y += _FPC_CHECK_((x+
#       1.25 ) *
#           10.0, 10, "compute.cu"); z = 
#                 _FPC_CHECK_(y + x, 11, "compute.cu");
#  return y;
#}

source = "compute.cu"
#expression1 = "(x + 1.25) * 10."
#expression2 = "y + x"

part1 = "y += _FPC_CHECK_((x+"
part2 = "10.0, 10, \"compute.cu\"); z = "
part3 = "_FPC_CHECK_(y + x, 11, \"compute.cu\");"

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make -f Makefile.0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
    cmd = ["make -f Makefile.0"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
   
    with open(source, 'r') as fd:
      lines = ''.join(fd.readlines())

    loc1 = lines.find(part1)
    loc2 = lines.find(part2)
    loc3 = lines.find(part3)
    assert loc1 < loc2
    assert loc2 < loc3

#    required_found1 = False
#    required_found2 = False
    
#    fd = open(source, "r")
#    for l in fd:
#        if "_FPC_CHECK_(" in l:
#            if expression1 in l:
#                loc1 = l.find("y +=")
#                loc2 = l.find("_FPC_CHECK_(")
#                loc3 = l.find(expression1)
#                loc4 = l.find(");")
#                assert loc1 < loc2
#                assert loc2 < loc3
#                assert loc3 < loc4
#                required_found1 = True
#                
#            if expression2 in l:
#                loc1 = l.find("_FPC_CHECK_(")
#                loc2 = l.find(expression2)
#                loc3 = l.find(");")
#                assert loc1 < loc2
#                assert loc2 < loc3
#                required_found2 = True
#                
#    fd.close()
    
#    assert required_found1
#    assert required_found2
