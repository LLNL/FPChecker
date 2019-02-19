#!/bin/bash -x

/Users/lagunaperalt1/projects/GPU_work/latest_llvm/llvm-08072018/install/bin/clang++ -c compute.cu -Xclang -load -Xclang /Users/lagunaperalt1/projects/fpchecker/code/tests/static/test_global_error_arr_found/../../../src/libcudakernels.so -include Runtime.h -I/Users/lagunaperalt1/projects/fpchecker/code/tests/static/test_global_error_arr_found/../../../src -O3  -x cuda --cuda-gpu-arch=sm_60 -g -emit-llvm -DFPC_ERRORS_DONT_ABORT
 /Users/lagunaperalt1/projects/GPU_work/latest_llvm/llvm-08072018/install/bin/llvm-dis -f compute-cuda-nvptx64-nvidia-cuda-sm_60.bc -o compute-cuda-nvptx64-nvidia-cuda-sm_60.ll
