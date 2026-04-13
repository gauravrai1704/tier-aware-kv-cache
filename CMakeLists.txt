cmake_minimum_required(VERSION 3.20)
project(tier_aware_kv_cache LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(USE_CUDA  "Build CUDA attention kernels"    OFF)
option(USE_RUST  "Build Rust AOL profiler"         OFF)
option(BUILD_TESTS "Build test executables"        ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Include dirs
# ---------------------------------------------------------------------------
include_directories(include)

# ---------------------------------------------------------------------------
# CUDA (optional)
# ---------------------------------------------------------------------------
if(USE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_definitions(-DUSE_CUDA)
    set(CUDA_SOURCES src/attention/paged_attention_kernel.cu)
endif()

# ---------------------------------------------------------------------------
# Core C++ library
# ---------------------------------------------------------------------------
set(CORE_SOURCES
    src/allocator/tiered_block_allocator.cpp
    src/migration/migration_orchestrator.cpp
)

if(USE_CUDA)
    list(APPEND CORE_SOURCES ${CUDA_SOURCES})
endif()

add_library(tierkv_core STATIC ${CORE_SOURCES})
target_include_directories(tierkv_core PUBLIC include)

if(USE_CUDA)
    target_link_libraries(tierkv_core PUBLIC CUDA::cudart CUDA::cuda_driver)
    set_target_properties(tierkv_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;90")
endif()

target_compile_options(tierkv_core PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -O3 -march=native>
    $<$<CXX_COMPILER_ID:MSVC>:/W4 /O2>
)

# ---------------------------------------------------------------------------
# Rust AOL profiler (optional)
# ---------------------------------------------------------------------------
if(USE_RUST)
    find_program(CARGO cargo REQUIRED)
    set(RUST_SRC_DIR ${CMAKE_SOURCE_DIR}/src/profiler)
    set(RUST_LIB ${CMAKE_BINARY_DIR}/libaol_profiler.a)

    add_custom_command(
        OUTPUT  ${RUST_LIB}
        COMMAND ${CARGO} build --release
                --manifest-path ${RUST_SRC_DIR}/Cargo.toml
                --target-dir    ${CMAKE_BINARY_DIR}/rust_build
        COMMAND ${CMAKE_COMMAND} -E copy
                ${CMAKE_BINARY_DIR}/rust_build/release/libaol_profiler.a
                ${RUST_LIB}
        DEPENDS ${RUST_SRC_DIR}/src/lib.rs
        COMMENT "Building Rust AOL profiler"
    )
    add_custom_target(aol_profiler_rust DEPENDS ${RUST_LIB})
    add_library(aol_profiler STATIC IMPORTED)
    set_target_properties(aol_profiler PROPERTIES
        IMPORTED_LOCATION ${RUST_LIB})
    add_dependencies(aol_profiler aol_profiler_rust)
    target_link_libraries(tierkv_core PUBLIC aol_profiler pthread dl)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/cpp/test_allocator.cpp")
        add_executable(test_allocator tests/cpp/test_allocator.cpp)
        target_link_libraries(test_allocator PRIVATE tierkv_core)
        add_test(NAME AllocatorTests COMMAND test_allocator)
    else()
        message(WARNING "tests/cpp/test_allocator.cpp not found — skipping")
    endif()

    if(EXISTS "${CMAKE_SOURCE_DIR}/tests/cpp/test_migration.cpp")
        add_executable(test_migration tests/cpp/test_migration.cpp)
        target_link_libraries(test_migration PRIVATE tierkv_core)
        add_test(NAME MigrationTests COMMAND test_migration)
    else()
        message(WARNING "tests/cpp/test_migration.cpp not found — skipping")
    endif()
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS tierkv_core
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/tierkv)
