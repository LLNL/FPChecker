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

def test_1():
    # --- compile code ---
    cmd = ["make"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ['./get_omp_lib_path.sh']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    # --- run code ---
    cmd = ["LD_LIBRARY_PATH="+cmdOutput.decode("utf-8")[:-1]+" ./main"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    found = False
    fileName = report.findReportFile('.fpc_logs')
    data = report.loadReport(fileName)
    for i in range(len(data)):
      print('i', i, data[i])
      if data[i]['file'].endswith('compute.cpp'):
        if data[i]['nan'] > 0:
          if data[i]['line'] == 24:
            found = True
            break

    assert found

if __name__ == '__main__':
    test_1()
