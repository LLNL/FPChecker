cmake_minimum_required(VERSION "3.9.1")
project(main_exe)

add_subdirectory(util)
add_subdirectory(core)

include_directories(util core)

set(main_source_files main.cpp)
add_executable(main_program ${main_source_files})

target_link_libraries(main_program server)
  # also transitively gets the "js" and "util" dependencies
