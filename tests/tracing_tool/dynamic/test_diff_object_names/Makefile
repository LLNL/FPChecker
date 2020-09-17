#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -arch=sm_70 -g

all:
	nvcc  $(OP) -c src/dot_product.cu -o src/dot_product_obj.cu.o
	nvcc  $(OP) -c src/main.cu -o src/main_obj.cu.o
	nvcc -o main src/dot_product_obj.cu.o src/main_obj.cu.o

clean:
	rm -rf src/*.o main __pycache__ src/*_copy.* main_fpc
