#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
OP  = -O0 -x cu -gencode=arch=compute_70,code=sm_70 -g
NVCC_OPTIONS=--ptxas-options=--verbose -Xptxas --device-debug --ptxas-options=-fmad=true -Xcompiler ,\"-fPIC\"

all:
	nvcc  $(OP) -c src/dot_product.cu $(NVCC_OPTIONS)
	nvcc  $(OP) -c src/main.cu
	nvcc -o main dot_product.o main.o

clean:
	rm -rf *.o main __pycache__ src/*_copy.* main_fpc
