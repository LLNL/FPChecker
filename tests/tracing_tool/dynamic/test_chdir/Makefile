#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -arch=sm_70 -g

all:
	cd src; nvcc  $(OP) -c dot_product.cu
	cd src; nvcc  $(OP) -c main.cu
	cd src; nvcc -o main dot_product.o main.o

clean:
	rm -rf src/*.o src/main __pycache__ src/*_copy.* src/main_fpc
