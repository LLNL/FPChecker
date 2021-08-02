
OP = 	-O2
CXX = FPC_INSTRUMENT=1 clang++-fpchecker 
CC =  FPC_INSTRUMENT=1 clang-fpchecker

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c compute.cpp $(OP)
	$(CC) -c compute_2.c $(OP)
	$(CXX) -o main compute.o compute_2.o main.o

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
