add_library(module OBJECT)

target_link_libraries(module PRIVATE yaml-cpp::yaml-cpp)

target_compile_definitions(module PRIVATE DEBUG)

target_sources(
    module PRIVATE

    module.cpp
    tensor.cpp
    module_graph.cpp
    timeboard.cpp
    activation.cpp
    layer.cpp
    layernorm.cpp
    parallel.cpp    
    linear.cpp
    attention.cpp
    compressed_kv_restore.cpp
    decoder.cpp
    communication.cpp
    embedding.cpp
    route.cpp
    rope.cpp
    residual.cpp
    expert.cpp
    lm_head.cpp
)

add_executable(timeboard-test test.cpp)

target_link_libraries(
    timeboard-test
    PRIVATE llm_system
)

set_target_properties(timeboard-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
