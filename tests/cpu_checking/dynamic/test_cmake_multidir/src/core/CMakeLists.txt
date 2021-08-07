
set(server_source_files
  server.cpp
  server.h
)
add_library(server ${server_source_files})

target_link_libraries(server util)
