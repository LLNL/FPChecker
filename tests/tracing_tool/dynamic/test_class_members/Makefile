#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -arch=sm_70 -g -std=c++11

all:
	nvcc 	$(OP) -c circle.cpp
	nvcc  $(OP) -c compute.cu
	nvcc  $(OP) -c main.cu
	nvcc -o main compute.o main.o circle.o

clean:
	rm -rf *.o main __pycache__ *_copy.* main_fpc
