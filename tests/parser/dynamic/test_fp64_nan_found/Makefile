
NVCC=FPC_INSTRUMENT=1 nvcc-fpc -std=c++11 -DFPC_ERRORS_ABORT

all:
	cp -f dot_product.cu.orig dot_product.cu
	cp -f main.cu.orig main.cu
	$(NVCC) -arch=sm_60 -O0 -c dot_product.cu
	$(NVCC) -arch=sm_60 -O0 -c main.cu
	$(NVCC) -o main dot_product.o main.o

clean:
	rm -rf *.cu *.o main __pycache__ *.ii .fpc_log.txt

