include ../../../Makefile.config

OP	= $(CLANG_PASS) -emit-llvm -O0 -x cuda --cuda-gpu-arch=sm_60 -g
#LINK	= -L${CUDA_PATH}/lib64/ -lcuda -lcudart
THRESHOLD       = -DFPC_DANGER_ZONE_PERCENT=0.20

REPORTS= -DFPC_ERRORS_DONT_ABORT

all:
	cp -f dot_product.cu.orig dot_product.cu
	cp -f main.cu.orig main.cu
	$(CC) -c dot_product.cu $(OP)
	$(CC) -c main.cu $(OP)
	nvcc $(INC_RUNTIME_PLUGIN) $(THRESHOLD) $(REPORTS) -arch=sm_60 -O0 -c dot_product.cu
	nvcc $(INC_RUNTIME_PLUGIN) $(THRESHOLD) $(REPORTS) -arch=sm_60 -O0 -c main.cu
	nvcc -o main dot_product.o main.o

clean:
	rm -rf *.cu *.o main __pycache__
