#!/usr/bin/env python

import subprocess
import os
import sys
from dynamic import report

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_command(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

def test_1():
    # --- compile code ---
    cmd = ["make"]
    run_command(cmd)

    # --- run code ---
    cmd = ["./main"]
    run_command(cmd)

    fileName = report.findHistogramFile('.fpc_logs')
    data = report.loadReport(fileName)

    assert data[0]['fp64']['0'] == 1
    assert data[0]['fp64']['1024'] == 2

    assert data[1]['fp64']['-1023'] == 1
    assert data[1]['fp64']['1024'] == 5
    
    assert data[2]['fp64']['0'] == 1
    assert data[2]['fp64']['1024'] == 2

    assert data[3]['fp64']['1024'] == 3
