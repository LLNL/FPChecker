
OP = 	-O2
CXX = clang++

all:
	$(CXX) -c main.cpp $(OP)
	$(CXX) -c compute.cpp $(OP)
	$(CXX) -o main compute.o main.o

clean:
	rm -rf *.o main __pycache__ .fpc_logs .fpc_log.txt
