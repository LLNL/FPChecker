#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -arch=sm_70 -g

all:
	nvcc  $(OP) -c dot_product.cu
	nvcc  $(OP) -c main.cu
	nvcc -o main dot_product.o  main.o

clean:
	rm -rf *.o main __pycache__ *_copy.* main_fpc fpchecker_conf.json
