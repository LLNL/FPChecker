include ../../Makefile.config

CUDA_PATH = /usr/tce/packages/cuda/cuda-9.2.148

#CC	=	/usr/workspace/wsa/laguna/latest_llvm/install/bin/clang++
CC	=	clang++

#LLVM_PASS       = -Xclang -load -Xclang /usr/workspace/wsa/laguna/fpchecker/FPChecker/src/libfpchecker.so \
-include Runtime.h -I/usr/workspace/wsa/laguna/fpchecker/FPChecker/src
#OP	= -arch=sm_60 -O0 

OP	= $(LLVM_PASS) -O0  -x cuda --cuda-gpu-arch=sm_60 -g
LINK	= -L${CUDA_PATH}/lib64/ -lcuda -lcudart

all:
	$(CC) -c compute.cu $(OP)
	$(CC) -c main.cu $(OP)
	$(CC) -o main compute.o main.o $(LINK)

clean:
	rm *.o main
