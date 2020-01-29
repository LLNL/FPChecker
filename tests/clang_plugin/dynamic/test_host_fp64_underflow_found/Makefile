include ../../../Makefile.config

OP	= $(CLANG_PASS) -emit-llvm -O0  -x cuda --cuda-gpu-arch=sm_60 -g

all:
	cp dot_product.cu.orig dot_product.cu
	cp main.cu.orig main.cu
	$(CC) $(OP) -c dot_product.cu
	$(CC) $(OP) -c main.cu
	#$(CC) -o main dot_product.o main.o $(LINK)
	nvcc $(INC_RUNTIME_PLUGIN) -O0 -arch=sm_60 -c dot_product.cu
	nvcc $(INC_RUNTIME_PLUGIN) -O0 -arch=sm_60 -c main.cu
	nvcc -o main dot_product.o main.o

clean:
	rm -rf *.o *.cu main __pycache__
