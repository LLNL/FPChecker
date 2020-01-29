#!/usr/bin/env python

import subprocess

def main():
    cmd = ["make -f Makefile.0"]
    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

    _fpc_functions_instrumented  = 0
    required_found = 0
    total_found = 0
    for l in cmdOutput.split("\n"):
        if "#FPCHECKER: Found " in l:
            total_found = total_found + 1
            if ("_FPC_DEVICE_CODE_FUNC_" in l or
                    "_FPC_PRINT_ERRORS_" in l or
                    "_FPC_FP32_CHECK_ADD_" in l or 
                    "_FPC_FP32_CHECK_SUB_" in l or
                    "_FPC_FP32_CHECK_MUL_" in l or
                    "_FPC_FP32_CHECK_DIV_" in l or
                    "_FPC_FP64_CHECK_ADD_" in l or
                    "_FPC_FP64_CHECK_SUB_" in l or
                    "_FPC_FP64_CHECK_MUL_" in l or
                    "_FPC_FP64_CHECK_DIV_" in l or
                    "_FPC_INTERRUPT_" in l or
                    "_FPC_WARNING_" in l or
                    "_FPC_PRINT_AT_MAIN_" in l):
                required_found = required_found +1
    
        # check if we are instrumenting any _FPC_ function
        if "#FPCHECKER: Instrumenting function:" in l:
            f = l.split()[3]
            if "_FPC_" in f:
                _fpc_functions_instrumented = _fpc_functions_instrumented + 1

    if (required_found==total_found and _fpc_functions_instrumented==0):
        print "PASSED"
    else:
        print "failed"

main()
