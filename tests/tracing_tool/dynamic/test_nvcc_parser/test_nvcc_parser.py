#!/usr/bin/env python

import subprocess
import os

PARSING_TOLL_PATH='../../../../tracing_tool/nvcc_parser.py'

NVCC_COMMANDS = [
  'nvcc -c compute.cu --x cu -I./ -Iinclude',
  'nvcc -c compute.cu -x=cu -I./ -Iinclude',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include --keep --dryrun',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include --link',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include -fatbin',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include -cubin',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include -dc',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include --device-c',
  'nvcc -c compute.cu -o=file.o --x cu -I./ -I=./include -g',
  'nvcc -c compute.cu  -I./ -I=./include -ftz true',
  'nvcc -c compute.cu  -I./ -I=./include -std c++11',
  'nvcc -c compute.cu  -I./ -I=./include -std=c++11',
  'nvcc -c compute.cu  -I./ -I=./include --relocatable-device-code=true',
  'nvcc -c compute.cu  -I./ -I=./include --relocatable-device-code false',
  'nvcc -c compute.cu  -I./ -I=./include --cudart=none',
  'nvcc -c compute.cu  -I./ -I=./include -ccbin gcc',
  'nvcc -c compute.cu  -I./ -I=./include -gencode arch=compute_60,code=compute_60',
  'nvcc -c compute.cu  -I./ -I=./include -arch sm_60',
  'nvcc -c compute.cu  -I./ -I=./include -Xptxas=-v,-O2',
  'nvcc -c compute.cu  -I./ -I=./include -DCOMP -D=COMP'
]

def setup_module(module):
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(THIS_DIR)

#def teardown_module(module):
    #cmd = ["make clean"]
    #cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)

def test_1():
    # --- compile code ---
    #cmd = [TRACING_TOLL_PATH + ' make']
    count = 0
    for cmd in NVCC_COMMANDS:
        try:
            print(cmd)
            cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            exit()
        count = count + 1

        try:
            nvcc_parser = 'python3 ' + PARSING_TOLL_PATH + ' "' + cmd + '\"'
            cmdOutput = subprocess.check_output(nvcc_parser, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            exit()

        parser_out = cmdOutput.decode('utf-8')[:-1]

        # Execute clang converted clang command
        try:
            print(parser_out)
            cmdOutput = subprocess.check_output(parser_out, stderr=subprocess.STDOUT, shell=True)
        except subprocess.CalledProcessError as e:
            print(e.output)
            exit()

 
    # --- run code ---
    #cmd = ["./main_fpc"]
    #try:
    #    cmdOutput = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    #except subprocess.CalledProcessError as e:
    #    print(e.output)
    #    exit()

    #rep = getFPCReport(cmdOutput.decode('utf-8').split("\n"))

    #assert rep[0] == 'NaN' and rep[3] == '8'
    assert count == len(NVCC_COMMANDS)
    
