#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -arch=sm_70 -g

all:
	nvcc  $(OP) -c src/dot_product.cu
	nvcc $(OP) -c src/cuda_launch.cu
	mpic++ -c -O2 src/main.cpp
	mpic++ -c -O2 src/mpi_stuff.cpp
	mpic++ -o main dot_product.o cuda_launch.o main.o mpi_stuff.o -L$(CUDA_HOME)/lib64 -lcudart

clean:
	rm -rf *.o main __pycache__ src/*_copy.* main_fpc
