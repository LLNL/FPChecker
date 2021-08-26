#!/usr/bin/env python

import subprocess
import os
import sys

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

def teardown_module(module):
    cmd = ["rm -rf .fpc_logs"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def run_command(cmd):
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print(e.output)
        exit()
    return cmdOutput

def test_1():
    cmd = ['cp -r traces .fpc_logs']
    run_command(cmd)
    assert os.path.isdir('.fpc_logs')

    cmd = ["fpc-create-report -c"]
    run_command(cmd)
    assert not os.path.isdir('.fpc_logs')
