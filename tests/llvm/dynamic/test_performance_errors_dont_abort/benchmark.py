#!/usr/bin/env python

import subprocess
import generateCode

#real	0m0.217s
#user	0m0.080s
#sys	0m0.113s

def getTime(lines):
    ret = 0
    for i in range(len(lines)):
        l = lines[i]
        if "real" in l:
            ret = l.split()[1]
            minu = ret.split("m")[0]
            secs = ret.split("m")[1][:-1]
            break

    return (minu, secs)

def runCode(i, k, b, app):
    nTimes = 10
    allTimes = []

    for i in range(nTimes):
        cmd = [" ".join(["time", app, str(i), str(k), str(b)])]
        try:
            cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print e.output
            exit()
        (minu, sec) = getTime(cmdOutput.split("\n"))
        totalTime = float(minu)*60.0 + float(sec)
        allTimes.append(totalTime)

    avgTime = sum(allTimes) / nTimes
    return avgTime

def buildCode():
    cmd = ["make"]
    try:
        cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    except subprocess.CalledProcessError as e:
        print e.output
        exit()


def main():
    numInst = [175]
#    numInst = [10, 50, 100, 125, 250, 375, 500, 750, 1000]
    kernelCalls = [10]
    blockSize = [128]

    #warm up
    runCode(10, 2, 32, "./main_base")
    runCode(10, 2, 32, "./main_errors_abort")
    runCode(10, 2, 32, "./main_errors_dont_abort")

    for i in numInst:
        for k in kernelCalls:
            for b in blockSize:
                generateCode.genCode(i)
                buildCode()

                t1 = runCode(i, k, b, "./main_base")
                #print "m:", minu, "s:", sec
                #t1 = float(minu)*60.0 + float(sec)

                t2 = runCode(i, k, b, "./main_errors_abort")
                #print "m:", minu, "s:", sec
                #t2 = float(minu)*60.0 + float(sec)

                t3 = runCode(i, k, b, "./main_errors_dont_abort")
                #print "m:", minu, "s:", sec
                #t3 = float(minu)*60.0 + float(sec) 

                over_e_abort = (t2/t1)
                over_e_dont_abort = (t3/t1)
                print "inst:", i, "calls:", k, "b_size:", b, "over_e_abort", over_e_abort, "over_e_dont_abort", over_e_dont_abort
main()

