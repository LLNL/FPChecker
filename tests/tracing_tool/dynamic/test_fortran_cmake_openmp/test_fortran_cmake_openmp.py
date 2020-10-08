#!/usr/bin/env python

import subprocess
import os
import sys
sys.path.append("..")
from .. import conftest

TRACING_TOLL_PATH = conftest.FPCHECKER_PATH

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["rm -rf build"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    return cmdOutput


def test_1():
    # --- compile code ---
    cmd = ['mkdir -p build && cd build && cmake ..']
    run(cmd)

    cmd = ['cd build && ' + TRACING_TOLL_PATH + ' --record make -j']
    run(cmd)

    cmd = ['cd build && make clean']
    run(cmd)

    cmd = ['cd build && ' + TRACING_TOLL_PATH + ' --inst-replay make -j']
    run(cmd)


    ## --- run code ---
    cmd = ["./build/src/main_program_fpc"]
    res = run(cmd)
    res = res.decode('utf-8').split('\n')

    #  Inside parallel section: N =         1109
    found = 0
    for line in res:
      if 'Inside parallel section' in line:
        found = found + 1

    assert found > 1

