#!/usr/bin/env python

import subprocess
import os

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make -f Makefile.0 clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)


# returns: tuple (error, op, file, line)
#
#+-------------------------- FPChecker Warning Report --------------------------+
# Error         : Underflow                                                      
# Operation     : ADD                                                            
# File          : dot_product.cu                                                 
# Line          : 9                                                              
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

def compileAndRun(op_level):
    # --- compile code ---
    cmd = ["make -f Makefile." + op_level]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["./main"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    ret = cmdOutput.decode('utf-8').split("\n")
    #print ret
    return ret

# Check we get exactly 64 errors in the report (8 elems x 8 threads)
def checkForErrorReports(out):
    ret = False
    firstLine = False
    secondLine = False
    thirdLine = False
    for l in out:
        if "#FPCHECKER: Warning at dot_product.cu:6" in l:
            firstLine = True
        if "#FPCHECKER: Warning at dot_product.cu:8" in l:
            secondLine = True
        if "#FPCHECKER: Warning at dot_product.cu:18" in l:
            thirdLine = True
    return (firstLine and secondLine and thirdLine)

def checkWarningsDontRepeat(out):
    cache = {}
    for l in out:
        if "#FPCHECKER: Warning at dot_product.cu:" in l:
            if l not in cache.keys():
                cache[l] = 1
            else:
                cache[l] = cache[l] + 1

    repeated = False
    for k in cache.keys():
        if cache[k] > 1:
            repeated = True
            break

    return (not repeated)

def test_1():
    op0_res = compileAndRun("0")
    rep0 = getFPCReport(op0_res)

    op1_res = compileAndRun("1")
    rep1 = getFPCReport(op1_res)

    op2_res = compileAndRun("2")
    rep2 = getFPCReport(op2_res)

    op3_res = compileAndRun("3")
    rep3 = getFPCReport(op3_res)

    assert rep0 == ("", "", "", "") and rep1 == ("", "", "", "") and rep2 == ("", "", "", "") and rep3 == ("", "", "", "")

    a1 = checkForErrorReports(op0_res)
    a2 = checkForErrorReports(op1_res)
    a3 = checkForErrorReports(op2_res)
    a4 = checkForErrorReports(op3_res)
    assert a1==True and a2==True and a3==True and a4==True

    b1 = checkWarningsDontRepeat(op0_res)
    b2 = checkWarningsDontRepeat(op1_res)
    b3 = checkWarningsDontRepeat(op2_res)
    b4 = checkWarningsDontRepeat(op3_res)
    assert b1==True and b2==True and b3==True and b4==True

