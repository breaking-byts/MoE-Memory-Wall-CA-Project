add_library(hardware OBJECT)

target_link_libraries(hardware PUBLIC yaml-cpp::yaml-cpp)

target_sources(
    hardware PRIVATE

    cluster.cpp
    node.cpp
    device.cpp
    executor.cpp
    
    layer_impl.cpp
    activation_impl.cpp
    linear_impl.cpp
    attention_sum_impl.cpp
    attention_gen_impl.cpp
    attention_mixed_impl.cpp
)

add_executable(hardware-test test.cpp)

target_link_libraries(
    hardware-test
    PRIVATE llm_system
)

set_target_properties(hardware-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)
