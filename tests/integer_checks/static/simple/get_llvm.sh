#!/bin/bash -x

/Users/lagunaperalt1/projects/GPU_work/latest_llvm/llvm-7.0/install/bin/clang++ -c -Xclang -load -Xclang /Users/lagunaperalt1/projects/fpchecker/code/tests/integer_checks/static/simple/../../../..//src/libfpchecker.so -include Runtime_int.h -I/Users/lagunaperalt1/projects/fpchecker/code/tests/integer_checks/static/simple/../../../..//src -O0 -g -emit-llvm comp.cpp 

/Users/lagunaperalt1/projects/GPU_work/latest_llvm/llvm-7.0/install/bin/llvm-dis -f comp.bc -o comp.ll
