include ../Makefile.config

PROG		= libfpchecker.so
SRC_FILES	= driver.cpp Utility.cpp Instrumentation.cpp CodeMatching.cpp Logging.cpp

PROG_CPU	    = libfpchecker_cpu.so
SRC_FILES_CPU	= driver_cpu.cpp Utility.cpp Instrumentation_cpu.cpp CodeMatching.cpp Logging.cpp

O_FILES		= $(SRC_FILES:%.cpp=%.o)
O_FILES_CPU = $(SRC_FILES_CPU:%.cpp=%.o)

all: $(PROG) $(PROG_CPU)

$(PROG): $(O_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROG) $(SHARED_LIB_OPT) $(O_FILES)

$(PROG_CPU): $(O_FILES_CPU)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROG_CPU) $(SHARED_LIB_OPT) $(O_FILES_CPU)

clean:
	$(RM) *.o $(PROG) $(PROG_CPU)
