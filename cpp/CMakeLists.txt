cmake_minimum_required(VERSION 3.10)
project(cp_ude CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wall -Wextra -O2 -ffp-contract=off)
include_directories(include)

add_executable(cp_ude_sim src/cp_ude_sim.cpp)
add_executable(test_alpha_filter src/test_alpha_filter.cpp)
target_compile_definitions(test_alpha_filter PRIVATE TESTDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/testdata")

enable_testing()
add_test(NAME alpha_filter_vs_rtl COMMAND test_alpha_filter)
