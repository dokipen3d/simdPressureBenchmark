cmake_minimum_required(VERSION 2.8)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS " -DNDEBUG -O3 -flto -march=native  -fopenmp  -fomit-frame-pointer")#-ffast-math
#set(CMAKE_CXX_FLAGS " -std=c++11 -O3 -flto -march=native -fomit-frame-pointer")

#INCLUDE_DIRECTORIES(/Users/dokipen/Documents/Projects/benchmark_apple_clang/include/)
INCLUDE_DIRECTORIES(/usr/local/include/ )
#link_directories(/Users/dokipen/Documents/Projects/benchmark_apple_clang/lib/ )
link_directories(/usr/local/lib/)
find_package(benchmark REQUIRED)
find_package(Boost)


project(fluidBenchmark2)
add_executable(${PROJECT_NAME} "main.cpp")
target_link_libraries(${PROJECT_NAME}  benchmark )
