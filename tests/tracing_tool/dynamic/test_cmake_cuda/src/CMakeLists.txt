cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cmake_and_cuda LANGUAGES CXX CUDA)

add_library(compute STATIC
  compute.cu
  compute.h
)

target_compile_features(compute PUBLIC cxx_std_11)

set_target_properties( compute 
                       PROPERTIES CUDA_SEPARABLE_COMPILATION ON
)


add_executable(main main.cu)

set_target_properties(main PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(main PRIVATE compute)

set(CMAKE_CUDA_FLAGS "-arch=sm_60 -Xptxas=-v")
