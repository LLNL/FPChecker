#!/bin/bash -x

clang++ -DFPC_ERRORS_DONT_ABORT -c compute.cu -Xclang -load -Xclang /Users/lagunaperalt1/projects/fpchecker/code/tests/static/test_simple_cuda/../../../build/libfpchecker.dylib -include Runtime.h -I/Users/lagunaperalt1/projects/fpchecker/code/tests/static/test_simple_cuda/../../../src -O3  -x cuda --cuda-gpu-arch=sm_60 -g -emit-llvm

llvm-dis -f compute-cuda-nvptx64-nvidia-cuda-sm_60.bc -o compute-cuda-nvptx64-nvidia-cuda-sm_60.ll
