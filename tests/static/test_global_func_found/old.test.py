#!/usr/bin/env python

import subprocess

def main():
    cmd = ["make"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

    passed = False
    for l in cmdOutput.split("\n"):
        #print l
        if "#FPCHECKER: Instrumenting function:" in l and "computed" in l:
            passed = True

    if passed == True:
        print "PASSED"
    else:
        print "failed"

main()
