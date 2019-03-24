#!/usr/bin/env python

import subprocess

def main():
    cmd = ["make"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

    passed = False
    ontargetFunc = False
    for l in cmdOutput.split("\n"):
        #print l
        if "#FPCHECKER: Instrumenting function:" in l and "powerd" in l:
            ontargetFunc = True
        
        if ontargetFunc == True:
            if "#FPCHECKER: Instrumented operations:" in l:
                val = int(l.split()[-1:][0])
                #print val
                if val == 6:
                    passed = True
                break

    if passed == True:
        print "PASSED"
    else:
        print "failed"

main()
