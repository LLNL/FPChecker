#!/usr/bin/env python

import subprocess
import os
#import sys
#sys.path.append("..")
#from .. import conftest

TRACING_TOLL_PATH = '../../../../tracing_tool/fpchecker'

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["make clean"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_exit(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()

def test_1():
    # --- compile code ---
    cmd = [TRACING_TOLL_PATH + ' --no-rollback make']
    failed = False
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        failed = True

    assert failed

    cmd = ['cp -f fpchecker_conf.json.orig fpchecker_conf.json']
    run_exit(cmd)

    cmd = [TRACING_TOLL_PATH + ' make']
    run_exit(cmd)
   
    cmd = ["./main_fpc"]
    run_exit(cmd) 


