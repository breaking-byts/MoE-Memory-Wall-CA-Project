add_executable(latency-test latency.cpp)
add_executable(sub-batch-test sub_batch_test.cpp)
add_executable(latency-export export_latency_trace.cpp)
add_executable(stat-export export_stat.cpp)
add_executable(stat-export-hetero export_stat_hetero.cpp)
add_executable(ablation_study ablation_study.cpp)
add_executable(run test.cpp)
add_executable(edap edap.cpp)
add_executable(sum_gen_split sum_gen_split.cpp)

target_link_libraries(
    latency-test
    PRIVATE llm_system
)

target_link_libraries(
    sum_gen_split
    PRIVATE llm_system
)

target_link_libraries(
    sub-batch-test
    PRIVATE llm_system
)

target_link_libraries(
    latency-export
    PRIVATE llm_system
)

target_link_libraries(
    ablation_study
    PRIVATE llm_system
)

target_link_libraries(
    stat-export
    PRIVATE llm_system
)

target_link_libraries(
    stat-export-hetero
    PRIVATE llm_system
)

target_link_libraries(
    run
    PRIVATE llm_system
)

target_link_libraries(
    edap
    PRIVATE llm_system
)


set_target_properties(latency-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(sum_gen_split PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)


set_target_properties(sub-batch-test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(latency-export PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(ablation_study PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(stat-export PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(stat-export-hetero PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(run PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

set_target_properties(edap PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)