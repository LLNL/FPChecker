#!/usr/bin/env python

import subprocess
import os
import sys
#sys.path.append('..')
#sys.path.append('.')
#import report
from dynamic import report

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
    # --- compile code ---
    cmd = ["make"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["mpirun -H localhost -np 4 --oversubscribe ./main"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    assert report.numberReportFiles('.fpc_logs') == 4

    found = False
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    for i in range(len(data)):
      print('i', i, data[i])
      if data[i]['file'].endswith('compute.cpp'):
        if data[i]['nan'] > 0:
          if data[i]['line'] == 10:
            found = True
            break

    assert found

