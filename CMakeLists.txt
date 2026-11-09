# Standalone (or add_subdirectory) build for header-only libkdtree + GoogleTest.
cmake_minimum_required(VERSION 3.20)

project(libkdtree
    VERSION 0.8.0
    DESCRIPTION "C++20 KD-tree header library with tests"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(libkdtree INTERFACE)
add_library(libkdtree::libkdtree ALIAS libkdtree)
target_compile_features(libkdtree INTERFACE cxx_std_20)
target_include_directories(libkdtree INTERFACE
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>")

# -----------------------------------------------------------------------------
# GoogleTest (FetchContent)
# -----------------------------------------------------------------------------
include(FetchContent)
set(LIBKDTREE_BUILD_TESTS ON CACHE BOOL "Build libkdtree unit tests")

if(LIBKDTREE_BUILD_TESTS)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()
    include(GoogleTest)

    add_executable(libkdtree_tests
        tests/test.cpp
        tests/test_large.cpp)
    target_link_libraries(libkdtree_tests PRIVATE
        libkdtree::libkdtree
        GTest::gtest
        GTest::gtest_main)
    target_compile_features(libkdtree_tests PRIVATE cxx_std_20)
    if(MSVC)
        # /bigobj: gtest + many template instantiations exceed the default section limit
        target_compile_options(libkdtree_tests PRIVATE /W4 /permissive- /bigobj)
    else()
        target_compile_options(libkdtree_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    # POST_BUILD discovery runs the exe from MSBuild without the gtest DLL dir on PATH
    # (shared CRT + gtest DLLs); PRE_TEST defers discovery to ctest when env is set up.
    gtest_discover_tests(libkdtree_tests
        DISCOVERY_MODE PRE_TEST
        PROPERTIES LABELS "libkdtree")
endif()
