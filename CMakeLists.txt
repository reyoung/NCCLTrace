cmake_minimum_required(VERSION 3.24)
project(nccltrace LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find zlib for gzip compression
find_package(ZLIB REQUIRED)
message(STATUS "Found ZLIB: ${ZLIB_LIBRARIES}")

add_subdirectory(3rd/catch2)
add_subdirectory(3rd/json)
add_library(nccltrace SHARED nccltrace.cpp
        lock_free_queue.h
        file_dumper.h
        dump_items.h
        comm_desc.h)

target_include_directories(nccltrace PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/3rd/json/single_include
)

target_link_libraries(nccltrace PRIVATE ZLIB::ZLIB)

enable_testing()
add_executable(nccltrace_lockfree_queue_test lock_free_queue_test.cpp)
target_link_libraries(nccltrace_lockfree_queue_test Catch2WithMain)
add_test(nccltrace_lockfree_queue_test nccltrace_lockfree_queue_test)

# FileDumper tests
add_executable(nccltrace_file_dumper_test file_dumper_test.cpp)
target_link_libraries(nccltrace_file_dumper_test Catch2WithMain ZLIB::ZLIB)
add_test(nccltrace_file_dumper_test nccltrace_file_dumper_test)

add_executable(nccltrace_comm_desc_test comm_desc_test.cpp)
target_link_libraries(nccltrace_comm_desc_test nccltrace Catch2WithMain)
add_test(nccltrace_comm_desc_test nccltrace_comm_desc_test)

# FreeRefPool tests
add_executable(nccltrace_free_reference_pool_test free_reference_pool_test.cpp)
target_link_libraries(nccltrace_free_reference_pool_test Catch2WithMain)
add_test(nccltrace_free_reference_pool_test nccltrace_free_reference_pool_test)

