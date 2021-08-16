
NVCC = nvcc-fpc

all:
	FPC_INSTRUMENT=1 $(NVCC) -std=c++11 -c -O0 -arch=sm_70 compute.cu

clean:
	rm -rf *.o *_inst.* *.ii .fpc_log.txt
