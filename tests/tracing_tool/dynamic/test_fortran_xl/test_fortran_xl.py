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
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)


def test_1():
    # --- compile code ---
    cmd = [TRACING_TOLL_PATH + ' --record make']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = ['make clean']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    cmd = [TRACING_TOLL_PATH + ' --inst-replay']
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()


    # --- run code ---
    cmd = ["./main_fpc"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

    rep = cmdOutput.decode('utf-8')

    assert "y =" in rep

