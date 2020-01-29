include ../../../Makefile.config

OP	= $(CLANG_PASS) -emit-llvm -O0  -x cuda --cuda-gpu-arch=sm_60 -g
LINK	= -L${CUDA_PATH}/lib64/ -lcuda -lcudart

all:
	cp -f dot_product.cu.orig dot_product.cu
	cp -f main.cu.orig main.cu
	$(CC) $(OP) -c dot_product.cu
	$(CC) $(OP) -c main.cu
	nvcc $(INC_RUNTIME_PLUGIN) -arch=sm_60 -O0 -c dot_product.cu
	nvcc $(INC_RUNTIME_PLUGIN) -arch=sm_60 -O0 -c main.cu
	nvcc -o main dot_product.o main.o

clean:
	rm -rf *.cu *.o main __pycache__
