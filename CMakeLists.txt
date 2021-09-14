cmake_minimum_required(VERSION 3.11)

set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")

project(fpchecker VERSION 0.2.0 DESCRIPTION "FPChecker" LANGUAGES CXX C)

execute_process(COMMAND llvm-config --ldflags 
OUTPUT_VARIABLE CMAKE_SHARED_LINKER_FLAGS 
OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --cxxflags
OUTPUT_VARIABLE CMAKE_CXX_FLAGS
OUTPUT_STRIP_TRAILING_WHITESPACE)

execute_process(COMMAND llvm-config --cppflags
OUTPUT_VARIABLE CMAKE_CPP_FLAGS 
OUTPUT_STRIP_TRAILING_WHITESPACE)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++14 -DFPC_DEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g -std=c++14")

include(GNUInstallDirs)
include(CheckLanguage)

###### Enable testing ##########
#set(ENV{PATH} "${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}")
enable_testing()
include(CTest)
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME cpu-tests COMMAND Python3::Interpreter -m pytest --tb=no
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/cpu_checking)
  set_tests_properties(cpu-tests PROPERTIES ENVIRONMENT "PATH=${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}")

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  add_test(NAME nvcc-tests COMMAND Python3::Interpreter -m pytest --tb=no
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/tests/parser)
  set_tests_properties(nvcc-tests PROPERTIES ENVIRONMENT "PATH=${CMAKE_INSTALL_PREFIX}/bin:$ENV{PATH}")

endif()
endif()
###############################

# Set Path pf wrappers in config file
set(NVCC_WRAPPER "${CMAKE_INSTALL_PREFIX}/bin/nvcc-fpchecker")
set(CLANG_WRAPPER "${CMAKE_INSTALL_PREFIX}/bin/clang-fpchecker")
set(CLANGPP_WRAPPER "${CMAKE_INSTALL_PREFIX}/bin/clang++-fpchecker")
set(MPI_WRAPPER "${CMAKE_INSTALL_PREFIX}/bin/mpicc-fpchecker")
set(MPIPP_WRAPPER "${CMAKE_INSTALL_PREFIX}/bin/mpicxx-fpchecker")
configure_file(interception_tool/intercept.h.in intercept.h)

add_library(fpchecker SHARED 
	src/CodeMatching.cpp
	src/Instrumentation.cpp
	src/Logging.cpp
	src/Utility.cpp
	src/driver.cpp
)

add_library(fpchecker_cpu SHARED 
	src/CodeMatching.cpp
	src/Instrumentation_cpu.cpp
	src/Logging.cpp
	src/Utility.cpp
	src/driver_cpu.cpp
)

add_library(fpchecker_plugin SHARED
	plugin/instrumentation_plugin.cpp
)

add_library(fpchecker_intercept_lib SHARED
	interception_tool/intercept.c
)

#set_target_properties(fpchecker_intercept_lib PROPERTIES LINKER_LANGUAGE C)
target_include_directories(fpchecker_intercept_lib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

if(APPLE)
	TARGET_LINK_LIBRARIES(fpchecker "-undefined dynamic_lookup")
	TARGET_LINK_LIBRARIES(fpchecker_cpu "-undefined dynamic_lookup")
	TARGET_LINK_LIBRARIES(fpchecker_plugin "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

TARGET_LINK_LIBRARIES(fpchecker_intercept_lib "-ldl")

set_target_properties(fpchecker PROPERTIES
    VERSION ${PROJECT_VERSION}
)

target_include_directories(fpchecker PRIVATE src)

set(CMAKE_INSTALL_DEFAULT_DIRECTORY_PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(TARGETS fpchecker fpchecker_plugin fpchecker_intercept_lib fpchecker_cpu
        LIBRARY DESTINATION "lib64"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install link for /lib
install(CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/lib64 ${CMAKE_INSTALL_PREFIX}/lib )"
        COMMENT "Creating link: /lib --> /lib64"
)

install(FILES "src/Runtime.h" "src/Runtime_plugin.h" "src/Runtime_parser.h" "src/Runtime_cpu.h" "src/FPC_Hashtable.h"
        DESTINATION "src"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ
)

install(FILES
	"interception_tool/fpchecker.py"
	"interception_tool/colors.py"
	DESTINATION interception 
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(FILES
        "parser/colors.py"
        "parser/config_reader.py"
        "parser/deprocess.py"
        "parser/exceptions.py"
        "parser/fpc-debug.py"
        "parser/fpc_logging.py"
        "parser/g++_fpchecker.py"
        "parser/instrument.py"
        "parser/match.py"
        "parser/nvcc_fpchecker.py"
        "parser/tokenizer.py"
        DESTINATION front-end
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(FILES
        "cpu_checking/cc_mpi_frontend.sh"
        "cpu_checking/cc_frontend.sh"
        "cpu_checking/cxx_mpi_frontend.sh"
        "cpu_checking/cxx_frontend.sh"
        "cpu_checking/clang_fpchecker.py"
        "cpu_checking/colors.py"
        "cpu_checking/exceptions.py"
        "cpu_checking/fpc_create_report.py"
        "cpu_checking/fpc_logging.py"
        "cpu_checking/line_highlighting.py"
        "cpu_checking/mpicc_fpchecker.py"
        DESTINATION cpu_checking
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

install(DIRECTORY "cpu_checking/report_templates/" DESTINATION cpu_checking/report_templates)

install(DIRECTORY
        DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        PATTERN "*"
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install link for fpchecker wrappers
install(CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/front-end/nvcc_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/nvcc-fpc )"
        COMMENT "Creating link: nvcc-fpc -> nvcc_fpchecker.py"
)

install(CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/front-end/nvcc_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/nvcc-fpchecker )"
        COMMENT "Creating link: nvcc-fpchecker -> nvcc_fpchecker.py"
)

# Install link for fpchecker reports tool
install(CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/front-end/fpc-debug.py ${CMAKE_INSTALL_PREFIX}/bin/fpc-report )"
        COMMENT "Creating link: fpc-report -> fpc-debug.py"
)

# Install link for fpchecker reports tool
install(CODE "execute_process( \
        COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/interception/fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/fpchecker )"
        COMMENT "Creating link: fpchecker -> fpchecker.py"
)

# Clang wrappers
#install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
#        ${CMAKE_INSTALL_PREFIX}/cpu_checking/clang_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/clang-fpchecker )"
#)

#install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
#        ${CMAKE_INSTALL_PREFIX}/cpu_checking/clang_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/clang++-fpchecker )"
#)

install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/cpu_checking/cc_frontend.sh ${CMAKE_INSTALL_PREFIX}/bin/clang-fpchecker )"
)

install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/cpu_checking/cxx_frontend.sh ${CMAKE_INSTALL_PREFIX}/bin/clang++-fpchecker )"
)

install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/cpu_checking/fpc_create_report.py ${CMAKE_INSTALL_PREFIX}/bin/fpc-create-report )"
)

#install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
#        ${CMAKE_INSTALL_PREFIX}/cpu_checking/mpicc_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/mpic++-fpchecker )"
#)

#install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
#        ${CMAKE_INSTALL_PREFIX}/cpu_checking/mpicc_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/mpicxx-fpchecker )"
#)

#install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
#        ${CMAKE_INSTALL_PREFIX}/cpu_checking/mpicc_fpchecker.py ${CMAKE_INSTALL_PREFIX}/bin/mpicc-fpchecker )"
#)

install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/cpu_checking/cxx_mpi_frontend.sh ${CMAKE_INSTALL_PREFIX}/bin/mpic++-fpchecker )"
)

install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/cpu_checking/cxx_mpi_frontend.sh ${CMAKE_INSTALL_PREFIX}/bin/mpicxx-fpchecker )"
)

install(CODE "execute_process( COMMAND ${CMAKE_COMMAND} -E create_symlink \
        ${CMAKE_INSTALL_PREFIX}/cpu_checking/cc_mpi_frontend.sh ${CMAKE_INSTALL_PREFIX}/bin/mpicc-fpchecker )"
)

# Tests
#file(COPY tests DESTINATION ${CMAKE_BINARY_DIR})
#file(COPY src/Runtime.h DESTINATION ${CMAKE_BINARY_DIR}/src)
#configure_file(tests/tests_Makefile.in tests/Makefile.config)
#configure_file(tests/llvm/static/_test_config.py.in tests/static/test_config.py)
#configure_file(tests/llvm/dynamic/_test_config.py.in tests/dynamic/test_config.py)
#add_custom_target(tests COMMAND bash ${CMAKE_BINARY_DIR}/tests/tests.sh)
#find_package(Python3 COMPONENTS Interpreter)

message(STATUS "CMAKE_CXX_COMPILE_FLAGS: " ${CMAKE_CXX_COMPILE_FLAGS})
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CPP_FLAGS: " ${CMAKE_CPP_FLAGS})
message(STATUS "CMAKE_SHARED_LINKER_FLAGS: " ${CMAKE_SHARED_LINKER_FLAGS})
message(STATUS "CMAKE_INSTALL_PREFIX: " ${CMAKE_INSTALL_PREFIX})
