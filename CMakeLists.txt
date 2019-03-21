cmake_minimum_required(VERSION 3.9)

project(fpchecker VERSION 0.1.0 DESCRIPTION "FPChecker" LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "clang++")

#find_package(LLVM  REQUIRED CONFIG)
#message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
#message(STATUS "LLVM_FOUND ${LLVM_FOUND}")
#message(STATUS "LLVM_DIR ${LLVM_DIR}")
#message(STATUS "LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
#message(STATUS "LLVM_DEFINITIONS: ${LLVM_DEFINITIONS}")
#message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
#message(STATUS "CMAKE_EXE_LINKER_FLAGS: ${CMAKE_EXE_LINKER_FLAGS}")

#include_directories(${LLVM_INCLUDE_DIRS})
#add_definitions(${LLVM_DEFINITIONS})

#COMMON_FLAGS    = -Wall -O3 -g -std=c++11
#LDFLAGS         += $(shell $(LLVM_CONFIG) --ldflags)
#CXXFLAGS        += $(COMMON_FLAGS) $(shell $(LLVM_CONFIG) --cxxflags) -fPIC -DFPC_DEBUG
#CPPFLAGS        += $(shell $(LLVM_CONFIG) --cppflags) -I$(SRC_DIR)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} `llvm-config --cxxflags --cppflags -Wall -O3 -g -std=c++11`")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++11 -DFPC_DEBUG")

#set(CMAKE_CPP_FLAGS "${CMAKE_CPP_FLAGS} `llvm-config --cppflags`")
#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} $(llvm-config --ldflags)")

execute_process(COMMAND llvm-config --ldflags 
OUTPUT_VARIABLE CMAKE_SHARED_LINKER_FLAGS 
OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --cxxflags
OUTPUT_VARIABLE CMAKE_CXX_FLAGS
OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --cppflags
OUTPUT_VARIABLE CMAKE_CPP_FLAGS 
OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++11 -DFPC_DEBUG")

include(GNUInstallDirs)

add_library(fpchecker SHARED 
	src/CodeMatching.cpp
	src/Instrumentation.cpp
	src/Logging.cpp
	src/Utility.cpp
	src/driver.cpp
)

if(APPLE)
	TARGET_LINK_LIBRARIES(fpchecker "-undefined dynamic_lookup")
endif()

set_target_properties(fpchecker PROPERTIES
    VERSION ${PROJECT_VERSION}
)

#configure_file(mylib.pc.in mylib.pc @ONLY)

target_include_directories(fpchecker PRIVATE src)

install(TARGETS fpchecker
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

#install(FILES ${CMAKE_BINARY_DIR}/mylib.pc
#    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)

message(STATUS "CMAKE_CXX_COMPILE_FLAGS: " ${CMAKE_CXX_COMPILE_FLAGS})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CPP_FLAGS: " ${CMAKE_CPP_FLAGS})
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS})
