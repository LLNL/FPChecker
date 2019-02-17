
CC	= $(LLVM_PATH)/bin/clang++

LLVM_PASS       = -Xclang -load -Xclang $(PWD)/../../../src/libcudakernels.so \
-include Runtime.h -I$(PWD)/../../../src

OP	= $(LLVM_PASS) -O0  -x cuda --cuda-gpu-arch=sm_60 -g

LINK	= -L${CUDA_PATH}/lib64/ -lcuda -lcudart

all:
	$(CC) -c compute.cu $(OP)

clean:
	rm *.o main
