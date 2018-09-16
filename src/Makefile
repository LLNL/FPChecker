include ../Makefile.config

PROG		= libcudakernels.so
#SRC_FILES	= driver.cpp Instrumentation.cpp Utility.cpp ProgressBar.cpp
SRC_FILES	= driver.cpp Utility.cpp Instrumentation.cpp CodeMatching.cpp

O_FILES		= $(SRC_FILES:%.cpp=%.o)

all: $(PROG)

$(PROG): $(O_FILES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $(PROG) $(SHARED_LIB_OPT) $(O_FILES)

clean:
	$(RM) *.o $(PROG)
