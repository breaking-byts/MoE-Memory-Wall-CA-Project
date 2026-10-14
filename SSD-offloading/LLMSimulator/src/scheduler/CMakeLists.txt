add_library(scheduler OBJECT)

target_sources(
    scheduler PRIVATE

    scheduler.cpp
    sequence.cpp
)

target_link_libraries(scheduler PRIVATE hardware)
# add_executable(scheduler-test test.cpp)

# target_link_libraries(
#     scheduler-test
#     PRIVATE llm_system
# )

# set_target_properties(scheduler-test PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
# )
