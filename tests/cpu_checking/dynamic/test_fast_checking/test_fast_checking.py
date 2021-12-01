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

def found_NaN(data):
    for i in range(len(data)):
      if data[i]['file'].endswith('compute.cpp'):
        if data[i]['nan'] > 0:
          return True
    return False

def found_cancellation(data):
    for i in range(len(data)):
      if data[i]['file'].endswith('compute.cpp'):
        if data[i]['cancellation'] > 0:
          return True
    return False

def found_comparison(data):
    for i in range(len(data)):
      if data[i]['file'].endswith('compute.cpp'):
        if data[i]['comparison'] > 0:
          return True
    return False

def test_1():
    cmd = ["make -f Makefile && ./main"]
    run_command(cmd)
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    assert found_NaN(data) and found_cancellation(data) and found_comparison(data)

    cmd = ["make clean && make -f Makefile_fast && ./main"]
    run_command(cmd)
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    assert found_NaN(data) and (not found_cancellation(data)) and (not found_comparison(data))

