include ../../Makefile.config

OP1	= -O0  -x cuda --cuda-gpu-arch=sm_60 -g
OP2	= $(LLVM_PASS) -O0  -x cuda --cuda-gpu-arch=sm_60 -g #-DFPC_ERRORS_DONT_ABORT -DFPC_DANGER_ZONE_PERCENT=0.80
OP3	= $(LLVM_PASS) -O0  -x cuda --cuda-gpu-arch=sm_60 -g -DFPC_ERRORS_DONT_ABORT


LINK	= -L${CUDA_PATH}/lib64/ -lcuda -lcudart

all:
	$(CC) -c arr_sum.cu $(OP1)
	$(CC) -c main.cu $(OP1)
	$(CC) -o main_base arr_sum.o main.o $(LINK)
	$(CC) -c arr_sum.cu $(OP2)
	$(CC) -c main.cu $(OP2)
	$(CC) -o main_errors_abort arr_sum.o main.o $(LINK)
	$(CC) -c arr_sum.cu $(OP3)
	$(CC) -c main.cu $(OP3)
	$(CC) -o main_errors_dont_abort arr_sum.o main.o $(LINK)

clean:
	rm -rf *.o main main_base main_errors_abort main_errors_dont_abort __pycache__
