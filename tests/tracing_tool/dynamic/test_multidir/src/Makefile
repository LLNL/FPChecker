#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -arch=sm_70 -g

all:
	cd dir1/dir2 && nvcc  $(OP) -c dot_product.cu && cd ../../../
	cd main && nvcc  $(OP) -c main.cu -I../dir1/dir2/ && cd ../dir1
	cd ~/
	nvcc -o main_program dir1/dir2/dot_product.o main/main.o

clean:
	rm -rf dir1/dir2/*.o main/*.o main_program __pycache__ *_copy.* main_program_fpc
