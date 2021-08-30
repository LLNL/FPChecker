
OP = 	-O2 
CXX = FPC_INSTRUMENT=1 clang++-fpchecker 
CC = FPC_INSTRUMENT=1 clang-fpchecker 

all:
	$(CXX) -c main_1.cpp $(OP)
	$(CXX) -o main_1 main_1.o
	$(CXX) -c main_2.cpp $(OP)
	$(CXX) -o main_2 main_2.o
	$(CXX) -c main_3.cpp $(OP)
	$(CXX) -o main_3 main_3.o
	$(CC) -c main_4.c $(OP)
	$(CC) -o main_4 main_4.o
	$(CC) -c main_5.c $(OP)
	$(CC) -o main_5 main_5.o
	$(CC) -c main_6.c $(OP)
	$(CC) -o main_6 main_6.o
	$(CC) -c main_7.c $(OP)
	$(CC) -o main_7 main_7.o

clean:
	rm -rf *.o main_1 main_2 main_3 main_4 main_5 main_6 main_7 __pycache__ .fpc_logs .fpc_log.txt
