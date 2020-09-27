cmake_minimum_required(VERSION "3.9.1")
project(main_exe)

add_subdirectory(compute)

include_directories(compute)

add_link_options(-fopenmp)

set(main_source_files main.c)
add_executable(main_program ${main_source_files})


target_link_libraries(main_program compute -lgfortran)
