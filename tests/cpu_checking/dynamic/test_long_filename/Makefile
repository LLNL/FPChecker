
OP = 	-O2 
CXX = FPC_INSTRUMENT=1 clang++-fpchecker 

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c compute_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long_long.cpp $(OP) -o compute.o
	$(CXX) -o main compute.o main.o

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
