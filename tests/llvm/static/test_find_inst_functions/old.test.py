#!/usr/bin/env python

import subprocess

def main():
	cmd = ["make"]
	cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

	count  = 0
	for l in cmdOutput.split("\n"):
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

	if count == 8:
		print "PASSED"
	else:
		print "failed"
main()
