#!/usr/bin/env python

import subprocess
import os
import sys
#sys.path.append('..')
#sys.path.append('.')
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

def run_and_check(cmd):
    run_command(cmd)
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    assert data[0]['input'] == cmd[0]+' '
    cmd = ["rm -rf .fpc_logs"]
    run_command(cmd)

def run_and_check_empty(cmd):
    run_command(cmd)
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    assert data[0]['input'] == ''
    cmd = ["rm -rf .fpc_logs"]
    run_command(cmd)

def test_1():
    # --- compile code ---
    cmd = ["make"]
    run_command(cmd)

    # --- run code ---
    cmd = ["./main_1 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check(cmd)
    cmd = ["./main_2 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check_empty(cmd)
    cmd = ["./main_3 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check(cmd)
    cmd = ["./main_4 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check(cmd)
    cmd = ["./main_5 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check_empty(cmd)
    cmd = ["./main_6 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check(cmd)
    cmd = ["./main_7 -o qqqq -o wwww -o eeee -o rrrr -o tttt -o yyyy"]
    run_and_check_empty(cmd)

