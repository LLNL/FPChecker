
enable_language(Fortran)

add_compile_options(-fopenmp)
add_link_options(-fopenmp)

set(compute_source_files
  compute.f95
)
add_library(compute ${compute_source_files})

target_link_libraries(compute)
