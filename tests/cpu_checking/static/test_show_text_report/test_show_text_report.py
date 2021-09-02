#!/usr/bin/env python

import subprocess
import os
import sys

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["rm -rf fpc-report"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_command(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()
    return cmdOutput

#[laguna@lassen709 test_show_text_report]$ fpc-create-report -s
#Generating FPChecker report...
#Trace files found: 1
#
#================== Main Report ===================
#positive_infinity              0
#negative_infinity              1
#nan                            8
#division_by_zero               4
#cancellation                   0
#comparison                     0
#underflow                      0
#latent_positive_infinity       0
#latent_negative_infinity       0
#latent_underflow               1

#[laguna@lassen709 test_show_text_report]$ fpc-create-report -s nan
#Generating FPChecker report...
#Trace files found: 1
#
#===== Nan Report =====
#./test.cpp:3
#./test.cpp:1
#
#===== Inputs =====
#./main


def test_1():
    cmd = ["fpc-create-report -s"]
    out = run_command(cmd).decode('utf-8')
    print(out)

    total = 0
    for l in out.split('\n'):
      if 'negative_infinity' in l:
        total += int(l.split()[1])
      if 'nan' in l:
        total += int(l.split()[1])
      if 'division_by_zero' in l:
        total += int(l.split()[1])
      if 'latent_underflow' in l:
        total += int(l.split()[1])

    assert total==14

    cmd = ["fpc-create-report -s nan"]
    out = run_command(cmd).decode('utf-8')

    total = 0
    for l in out.split('\n'):
      if './test.cpp:3' in l or './test.cpp:1' in l or './main' in l:
        total += 1

    assert total==3
