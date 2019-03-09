include ../Makefile.config

PROG		= libfpchecker.so
SRC_FILES	= driver.cpp Utility.cpp Instrumentation.cpp CodeMatching.cpp Logging.cpp

O_FILES		= $(SRC_FILES:%.cpp=%.o)

all: $(PROG)

$(PROG): $(O_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROG) $(SHARED_LIB_OPT) $(O_FILES)

clean:
	$(RM) *.o $(PROG)
