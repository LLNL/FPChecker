
enable_language(Fortran)

set(compute_source_files
  compute.f95
)
add_library(compute ${compute_source_files})

target_link_libraries(compute)
