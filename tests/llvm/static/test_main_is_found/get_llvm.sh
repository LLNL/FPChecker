#!/bin/bash -x

clang++ -emit-llvm -c compute.cu \
-include Runtime.h -I/Users/lagunaperalt1/projects/fpchecker/code/tests/llvm/static/test_main_is_found/../../../../src -O0  -x cuda --cuda-gpu-arch=sm_60 -g
llvm-dis -f compute.bc -o compute.ll
