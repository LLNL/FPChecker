#include ../../../Makefile.config

#OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
#LINK	= -L${CUDA_PATH}/lib64/ -lcuda -lcudart

NVCC = /usr/workspace/wsa/laguna/fpchecker/FPChecker/parser/nvcc_fpchecker.py
#NVCC = nvcc

INCLUDE = -include /usr/workspace/wsa/laguna/fpchecker/FPChecker/src/Runtime_parser.h

all:
	cp -f dot_product.cu.orig dot_product.cu
	cp -f main.cu.orig main.cu
	#nvcc  -arch=sm_70 -O0 -E dot_product.cu -o dot_product.cu.ii
	#nvcc  -arch=sm_70 -O0 -E main.cu -o main.cu.ii
	#python3 ../instrument.py dot_product.cu.ii dot_product.cu
	#python3 ../instrument.py main.cu.ii main.cu
	#nvcc $(INCLUDE) -arch=sm_70 -O0 -c dot_product_inst.cu 
	#nvcc $(INCLUDE) -arch=sm_70 -O0 -c main_inst.cu 
	$(NVCC) --std=c++11 -arch=sm_70 -O0 -c dot_product.cu -o dot_product.cu.o
	$(NVCC) --std=c++11 -arch=sm_70 -O0 -c main.cu  -o main.cu.o
	$(NVCC) -o main main.cu.o dot_product.cu.o

clean:
	rm -rf *.cu *.o main __pycache__ *_inst.* *.inst *.ii
