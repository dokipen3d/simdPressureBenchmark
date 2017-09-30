cmake_minimum_required(VERSION 2.8)


find_package(benchmark)

project(fluidBenchmark)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME} benchmark)
