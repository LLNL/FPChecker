#!/usr/bin/env python

import subprocess
import os

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean"]
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

def test_1():
    # --- compile code ---
    cmd = ["make"]
    try:
        cmdOutput_all = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["./main"]
    try:
        cmdOutput_all = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ["make -f Makefile.no_subnormal"]
    try:
        cmdOutput_no_sub = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["./main"]
    try:
        cmdOutput_no_sub = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ["make -f Makefile.no_warnings"]
    try:
        cmdOutput_no_war = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["./main"]
    try:
        cmdOutput_no_war = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ["make -f Makefile.no_warnings_and_subnormal"]
    try:
        cmdOutput_no_war_no_sub = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["./main"]
    try:
        cmdOutput_no_war_no_sub = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    nan_error = 0
    inf_error = 0
    sub_error = 0
    sub_war = 0
    inf_war = 0
    for l in cmdOutput_all.decode('utf-8').split("\n"):
        if "#FPCHECKER: NaN error:" in l:
            nan_error = 1
        if "#FPCHECKER: INF error:" in l:
            inf_error = 1
        if "#FPCHECKER: underflow error:" in l:
            sub_error = 1
        if "#FPCHECKER: underflow warning:" in l:
            sub_war = 1
        if "#FPCHECKER: overflow warning:" in l:
            inf_war = 1

    assert nan_error == 1 and inf_error == 1 and sub_error == 1 and sub_war == 1 and inf_war == 1

    nan_error = 0
    inf_error = 0
    sub_error = 0
    sub_war = 0
    inf_war = 0
    for l in cmdOutput_no_sub.decode('utf-8').split("\n"):
        if "#FPCHECKER: NaN error:" in l:
            nan_error = 1
        if "#FPCHECKER: INF error:" in l:
            inf_error = 1
        if "#FPCHECKER: underflow error:" in l:
            sub_error = 1
        if "#FPCHECKER: underflow warning:" in l:
            sub_war = 1
        if "#FPCHECKER: overflow warning:" in l:
            inf_war = 1

    assert nan_error == 1 and inf_error == 1 and sub_error == 0 and sub_war == 1 and inf_war == 1

    nan_error = 0
    inf_error = 0
    sub_error = 0
    sub_war = 0
    inf_war = 0
    for l in cmdOutput_no_war.decode('utf-8').split("\n"):
        if "#FPCHECKER: NaN error:" in l:
            nan_error = 1
        if "#FPCHECKER: INF error:" in l:
            inf_error = 1
        if "#FPCHECKER: underflow error:" in l:
            sub_error = 1
        if "#FPCHECKER: underflow warning:" in l:
            sub_war = 1
        if "#FPCHECKER: overflow warning:" in l:
            inf_war = 1

    assert nan_error == 1 and inf_error == 1 and sub_error == 1 and sub_war == 0 and inf_war == 0

    nan_error = 0
    inf_error = 0
    sub_error = 0
    sub_war = 0
    inf_war = 0
    for l in cmdOutput_no_war_no_sub.decode('utf-8').split("\n"):
        if "#FPCHECKER: NaN error:" in l:
            nan_error = 1
        if "#FPCHECKER: INF error:" in l:
            inf_error = 1
        if "#FPCHECKER: underflow error:" in l:
            sub_error = 1
        if "#FPCHECKER: underflow warning:" in l:
            sub_war = 1
        if "#FPCHECKER: overflow warning:" in l:
            inf_war = 1

    assert nan_error == 1 and inf_error == 1 and sub_error == 0 and sub_war == 0 and inf_war == 0


