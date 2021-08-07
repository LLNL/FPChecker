OP  = -O0 -x cu -arch=sm_70 -g -std=c++11 -DFPC_ERRORS_ABORT

all:
	nvcc  $(OP) -c dot_product.cu
	nvcc  $(OP) -c main.cu
	nvcc -o main dot_product.o main.o

clean:
	rm -rf *.o main __pycache__ *_copy.* main_fpc *_inst.* *.ii .fpc_log.txt
