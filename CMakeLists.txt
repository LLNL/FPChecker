cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")

project(fpchecker VERSION 0.1.0 DESCRIPTION "FPChecker" LANGUAGES CXX)

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

target_include_directories(fpchecker PRIVATE src)

install(TARGETS fpchecker
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES src/Runtime.h 
        DESTINATION src
)

# Tests
file(COPY tests DESTINATION ${CMAKE_BINARY_DIR})
file(COPY src/Runtime.h DESTINATION ${CMAKE_BINARY_DIR}/src)
configure_file(tests/tests_Makefile.in tests/Makefile.config)
configure_file(tests/static/test_config.py.in tests/static/test_config.py)
configure_file(tests/dynamic/test_config.py.in tests/dynamic/test_config.py)
#enable_testing()
#add_test(NAME static_tests COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/tests/static/test.py)
#add_test(NAME dynamic_tests COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_BINARY_DIR}/tests/dynamic/test.py)
add_custom_target(tests COMMAND bash ${CMAKE_BINARY_DIR}/tests/tests.sh)

message(STATUS "CMAKE_CXX_COMPILE_FLAGS: " ${CMAKE_CXX_COMPILE_FLAGS})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CPP_FLAGS: " ${CMAKE_CPP_FLAGS})
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS})
