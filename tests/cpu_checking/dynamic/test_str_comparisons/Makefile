
OP = 	-O2 
CXX = FPC_INSTRUMENT=1 clang++-fpchecker 

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -o main main.o

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
