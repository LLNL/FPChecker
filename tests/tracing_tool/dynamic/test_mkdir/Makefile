#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -arch=sm_70 -g

all: dot_product.o main.o
	cd build; nvcc -o main dot_product.o main.o

main.o: src/main.cu
	mkdir -p build; cp src/main.cu build/
	cd build; nvcc  $(OP) -c main.cu

dot_product.o: src/dot_product.cu
	mkdir -p build; cp src/dot_product.cu build/; cp src/dot_product.h build/
	cd build; nvcc  $(OP) -c dot_product.cu

clean:
	rm -rf build __pycache__ 
