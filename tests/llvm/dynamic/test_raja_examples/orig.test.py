#!/usr/bin/env python

import subprocess

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

def main():
    # --- compile code ---
    cmd = ["make -f Makefile.errors_abort"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print e.output
	exit()

    # --- run code ---
    cmd = ["./main"]
    error = False
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        cmdOutput =  e.output
        #exit()
        error = True

    rep = getFPCReport(cmdOutput.split("\n"))

    # --- compile code ---
    cmd = ["make -f Makefile.errors_dont_abort"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print e.output
        exit()

    # --- run code ---
    cmd = ["./main"]
    error = False
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        cmdOutput =  e.output
        #exit()
        error = True

    error_no_aborts = False
    for l in cmdOutput.split("\n"):
        if "#FPCHECKER: Underflow Error at dot_product_raja.cpp:32" in l:
            error_no_aborts = True

    if rep[0] == 'Underflow' and rep[3] == '32' and error_no_aborts == True:
        print "PASSED"
    else:
        print "failed"

main()

