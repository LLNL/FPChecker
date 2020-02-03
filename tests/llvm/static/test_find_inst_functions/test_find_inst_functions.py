#!/usr/bin/env python

import subprocess
import os

def setup_module(module):
	THIS_DIR = os.path.dirname(os.path.abspath(__file__))
	os.chdir(THIS_DIR)

def teardown_module(module):
	cmd = ["make clean"]
	cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
	cmd = ["make"]
	cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

	count  = 0
	for l in cmdOutput.decode('utf-8').split("\n"):
		if "#FPCHECKER: Found _FPC_FP32_CHECK_ADD_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP32_CHECK_SUB_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP32_CHECK_MUL_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP32_CHECK_DIV_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP64_CHECK_ADD_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP64_CHECK_SUB_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP64_CHECK_MUL_" in l:
			count = count + 1
		if "#FPCHECKER: Found _FPC_FP64_CHECK_DIV_" in l:
			count = count + 1

	assert count == 8
