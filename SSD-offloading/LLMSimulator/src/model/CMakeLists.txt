add_library(model OBJECT)

target_sources(
    model PRIVATE
    
    llm.cpp   
)

add_executable(model-test test.cpp)

target_link_libraries(
    model-test
    PRIVATE llm_system
)

set_target_properties(model-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

target_link_libraries(model PRIVATE hardware)
