add_library(dram STATIC)

add_subdirectory(pimkernel)
add_subdirectory(ramulator2)
#add_dependencies(dram ramulator)

target_sources(
  dram PRIVATE

  dram_interface.cpp
  dram_request.cpp
  pim_request.cpp
  memory_object.cpp
  mmap_controller.cpp
)

# target_link_libraries(
#   dram 
#   PRIVATE ramulator
#   PRIVATE pimkernel
# )

add_executable(dram-test)
target_sources(
  dram-test
  PRIVATE test.cpp
)

target_link_libraries(
  dram-test
  PRIVATE llm_system
)

set_target_properties(
  dram-test
  PROPERTIES
  OUTPUT_NAME dram-test
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

