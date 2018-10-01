#include "benchmark/benchmark.h"
#include <vector>
#include <array>
#include <algorithm>
#include <iostream>
#include <cstddef>
#include <boost/align/aligned_allocator.hpp>

#include <x86intrin.h>

constexpr std::size_t dimension8 = 8;
constexpr std::size_t dimension8Squared = dimension8*dimension8;
constexpr std::size_t dimension8Cubed = dimension8*dimension8*dimension8;

constexpr std::size_t dimension12 = 12;
constexpr std::size_t dimension12Squared = dimension12*dimension12;
constexpr std::size_t dimension12Cubed = dimension12*dimension12*dimension12;


constexpr std::size_t dimension512 = 1024;
constexpr std::size_t dimension512Squared = dimension512*dimension512;

//512^2 4*32 was good
//1024^2 4*16 & 4*64 was good
//2048^2 chunking by 4x16 chunks was good


//inline static int coord3to1(int x, int y, int z, int width){
//    return (x * width + y) * width + z;
//}

//__attribute__ ((always_inline))
//inline static std::size_t coord3to1(int x, int y, int z, int width){
//    return (x +  width * ( y + width * z));
//}

__attribute__ ((always_inline))
inline static std::size_t coord3to1(std::size_t x, std::size_t y, std::size_t z, std::size_t width){
    return (z +  width * ( y + width * x));
}

//__attribute__ ((always_inline))
//inline static std::size_t coord3to1Edge(int y, int z, int width){
//    return (width * ( y + width * z));
//}


__attribute__ ((always_inline))
inline static std::size_t coord3to1Edge(std::size_t y, std::size_t z, std::size_t width){
    return (width * ( y + width * z));
}


__attribute__ ((always_inline))
inline static std::size_t coord2dto1Inv(std::size_t x, std::size_t y, std::size_t width){
    //return (width * ( y + width * z));
    //return y* width + z;
    return y + x * width;

}

__attribute__ ((always_inline))
inline static std::size_t coord2dto1(std::size_t x, std::size_t y, std::size_t width){
    //return (width * ( y + width * z));
    //return y* width + z;
    return x + y * width;
}

template <int N>
inline static std::size_t coord2dto1T(std::size_t y, std::size_t z){
    //return (width * ( y + width * z));
    //return y* width + z;
    return y + z* N;

}
//inline static std::size_t coord3to1(int x, int y, int z){
//    return (x +  dimension8 * ( y + dimension8 * z));
//}
//inline static std::size_t constexpr coord3to1(std::size_t x, std::size_t y, std::size_t z, std::size_t width){
//    return (x +  y * width + z * width * width);
//}

static void pressurePaddedSingle(benchmark::State& state) {

    //fromVec8.clear();
    //std::fill(begin(fromVec8), end(fromVec8), 1);
    auto fromVec8 = std::vector<float>(dimension8*dimension8*dimension8);
    auto toVec8 = std::vector<float>(dimension8*dimension8*dimension8);
    auto divVec8 = std::vector<float>(dimension8*dimension8*dimension8);



    while (state.KeepRunning()){
        for (std::size_t k = 1; k < dimension8-1; ++k){
            for (std::size_t j = 1; j < dimension8-1; ++j){
                for (std::size_t i = 1; i < dimension8-1; ++i){
                       const std::size_t index = coord3to1(i, j, k, dimension8);

                      toVec8[index] =  (fromVec8[index-1] +
                                        fromVec8[index+1] +
                                        fromVec8[index+dimension8] +
                                        fromVec8[index-dimension8] +
                                        fromVec8[index+dimension8Squared] +
                                        fromVec8[index-dimension8Squared] +
                                        divVec8[index])/6.0f;


                }
            }
        }
        for (std::size_t k = 2; k < dimension8-2; ++k){
            for (std::size_t j = 2; j < dimension8-2; ++j){
                for (std::size_t i = 2; i < dimension8-2; ++i){
                       const std::size_t index = coord3to1(i, j, k, dimension8);

                       fromVec8[index] =  (toVec8[index-1] +
                                        toVec8[index+1] +
                                        toVec8[index+dimension8] +
                                        toVec8[index-dimension8] +
                                        toVec8[index+dimension8Squared] +
                                        toVec8[index-dimension8Squared] +
                                        divVec8[index])/6.0f;


                }
            }
        }
        benchmark::DoNotOptimize(toVec8.data());
        benchmark::DoNotOptimize(fromVec8.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.

    }


}
// Register the function as a benchmark
BENCHMARK(pressurePaddedSingle);

static void pressurePaddedSingleArrayCast(benchmark::State& state) {

    //fromVec8.clear();
    //std::fill(begin(fromVec8), end(fromVec8), 1);
    auto fromVec8 = std::vector<float>(dimension8*dimension8*dimension8);
    auto toVec8 = std::vector<float>(dimension8*dimension8*dimension8);
    auto divVec8 = std::vector<float>(dimension8*dimension8*dimension8);

    float(&fromArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(fromVec8.data());
    float(&toArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(toVec8.data());
    float(&divArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(divVec8.data());

    while (state.KeepRunning()){
        for (std::size_t k = 1; k < dimension8-1; ++k){
            for (std::size_t j = 1; j < dimension8-1; ++j){
                for (std::size_t i = 1; i < dimension8-1; ++i){


                    toArray[i][j][k] =  (fromArray[i-1][j][k] +
                                        fromArray[i+1][j][k] +
                                        fromArray[i][j+1][k] +
                                        fromArray[i][j-1][k] +
                                        fromArray[i][j][k+1] +
                                        fromArray[i][j][k-1] +
                                        divArray[i][j][k])/6.0f;


                }
            }
        }
        for (std::size_t k = 2; k < dimension8-2; ++k){
            for (std::size_t j = 2; j < dimension8-2; ++j){
                for (std::size_t i = 2; i < dimension8-2; ++i){
                       const std::size_t index = coord3to1(i, j, k, dimension8);

                       fromArray[i][j][k] =   (toArray[i-1][j][k] +
                               toArray[i+1][j][k] +
                               toArray[i][j+1][k] +
                               toArray[i][j-1][k] +
                               toArray[i][j][k+1] +
                               toArray[i][j][k-1] +
                               divArray[i][j][k])/6.0f;



                }
            }
        }
        benchmark::DoNotOptimize(toVec8.data());
        benchmark::DoNotOptimize(fromVec8.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.

    }


}
// Register the function as a benchmark
BENCHMARK(pressurePaddedSingleArrayCast);


static void pressurePaddedSingleArray(benchmark::State& state) {

    //fromVec8.clear();
    //std::fill(begin(fromVec8A), end(fromVec8A), 1);

    while (state.KeepRunning()){
        std::array<float,dimension8Cubed> fromVec8A;
        std::array<float,dimension8Cubed> toVec8A;
        std::array<float,dimension8Cubed> divVec8A;
        for (int k = 1; k < dimension8-1; ++k){
            for (int j = 1; j < dimension8-1; ++j){
                for (int i = 1; i < dimension8-1; ++i){
                       const int index = coord3to1(i, j, k, dimension8);

                       toVec8A[index] =  (fromVec8A[index-1] +
                                        fromVec8A[index+1] +
                                        fromVec8A[index+dimension8] +
                                        fromVec8A[index-dimension8] +
                                        fromVec8A[index+dimension8Squared] +
                                        fromVec8A[index-dimension8Squared] +
                                        divVec8A[index])/6.0f;


                }
            }
        }
        for (int k = 2; k < dimension8-2; ++k){
            for (int j = 2; j < dimension8-2; ++j){
                for (int i = 2; i < dimension8-2; ++i){
                       const int index = coord3to1(i, j, k, dimension8);

                       fromVec8A[index] =  (toVec8A[index-1] +
                                        toVec8A[index+1] +
                                        toVec8A[index+dimension8] +
                                        toVec8A[index-dimension8] +
                                        toVec8A[index+dimension8Squared] +
                                        toVec8A[index-dimension8Squared] +
                                        divVec8A[index])/6.0f;


                }
            }
        }
        benchmark::DoNotOptimize(toVec8A.data());

        benchmark::DoNotOptimize(fromVec8A.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.

    }

}
//BENCHMARK(pressurePaddedSingleArray);

//4 padded to 8 - get 2 iterations done
static void pressurePaddedsimd8(benchmark::State& state) {


    //std::fill(begin(fromVec8), end(fromVec8), 1);

    auto fromVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    auto toVec8 = std::vector<float, boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    auto divVec8 = std::vector<float, boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    float(&fromArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(fromVec8.data());
    float(&toArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(toVec8.data());
    float(&divArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(divVec8.data());

    __m256 six6 = _mm256_set1_ps(0.16666666f);

    while (state.KeepRunning()){


        for (std::size_t k = 1; k < dimension8-1; ++k){

            for (std::size_t j = 1; j < dimension8-1; ++j){

               const std::size_t index = coord3to1Edge(j, k, dimension8);

               __m256 U = _mm256_load_ps(&fromVec8[index+dimension8]);
               __m256 D = _mm256_load_ps(&fromVec8[index-dimension8]);
               __m256 F = _mm256_load_ps(&fromVec8[index+dimension8Squared]);
               __m256 B = _mm256_load_ps(&fromVec8[index-dimension8Squared]);

               //right shift
               __m256 R_a = _mm256_load_ps(&fromVec8[index]);
               //a7, a6, a5, a4, a3, a2, a1, a0 -> a6, a5, a4, a3, a2, a1, a0, 0
               __m256 p0 = _mm256_permute_ps(R_a, 0x93); //10010011=0x93
               //p0 = a6, a5, a4, a7, a2, a1, a0, a3
               __m256 p1 = _mm256_permute2f128_ps(p0, p0, 0x8); //00001000=0x8
               //p1 = a2, a1, a0, a3, 0, 0, 0, 0
               __m256 R = _mm256_blend_ps(p0, p1, 0x11); //p2
               //p2 = a6, a5, a4, a3, a2, a1, a0, 0

               //Left shift
               __m256 p0l = _mm256_permute_ps(R_a, 0x39); //00111001=0x39
               //p0 = a4, a7, a6, a5, a0, a3, a2, a1
               __m256 p1l = _mm256_permute2f128_ps(p0l, p0l, 0x81);//10000001=0x81
               //p1 = 0, 0, 0, 0, a4, a7, a6, a5
               __m256 L = _mm256_blend_ps(p0l, p1l, 0x88); //10001000=0x88 p2
               //p2 = 0, a7, a6, a5, a4, a3, a2, a1


               __m256 finA = _mm256_add_ps(L, R);
               __m256 finB = _mm256_add_ps(U, D);
               __m256 finC = _mm256_add_ps(F, B);

               __m256 finD = _mm256_add_ps(finA, finB);
               __m256 finE = _mm256_add_ps(finC, finD);
               __m256 Div = _mm256_load_ps(&divVec8[index]);

               __m256 finF = _mm256_add_ps(finE, Div);

               __m256 final = _mm256_mul_ps(finF, six6);

               _mm256_store_ps(&toVec8[index], final);


            }
        }
        for (std::size_t k = 2; k < dimension8-2; ++k){
            for (std::size_t j = 2; j < dimension8-2; ++j){

               const std::size_t index = coord3to1Edge(j, k, dimension8);

               __m256 U = _mm256_load_ps(&toVec8[index+dimension8]);
               __m256 D = _mm256_load_ps(&toVec8[index-dimension8]);
               __m256 F = _mm256_load_ps(&toVec8[index+dimension8Squared]);
               __m256 B = _mm256_load_ps(&toVec8[index-dimension8Squared]);

               //right shift
               __m256 R_a = _mm256_load_ps(&toVec8[index]);
               //a7, a6, a5, a4, a3, a2, a1, a0 -> a6, a5, a4, a3, a2, a1, a0, 0
               __m256 p0 = _mm256_permute_ps(R_a, 0x93); //10010011=0x93
               //p0 = a6, a5, a4, a7, a2, a1, a0, a3
               __m256 p1 = _mm256_permute2f128_ps(p0, p0, 0x8); //00001000=0x8
               //p1 = a2, a1, a0, a3, 0, 0, 0, 0
               __m256 R = _mm256_blend_ps(p0, p1, 0x11); //p2
               //p2 = a6, a5, a4, a3, a2, a1, a0, 0

               //Left shift
               __m256 p0l = _mm256_permute_ps(R_a, 0x39); //00111001=0x39
               //p0 = a4, a7, a6, a5, a0, a3, a2, a1
               __m256 p1l = _mm256_permute2f128_ps(p0l, p0l, 0x81);//10000001=0x81
               //p1 = 0, 0, 0, 0, a4, a7, a6, a5
               __m256 L = _mm256_blend_ps(p0l, p1l, 0x88); //10001000=0x88 p2
               //p2 = 0, a7, a6, a5, a4, a3, a2, a1

               __m256 finA = _mm256_add_ps(L, R);
               __m256 finB = _mm256_add_ps(U, D);
               __m256 finC = _mm256_add_ps(F, B);

               __m256 finD = _mm256_add_ps(finA, finB);
               __m256 finE = _mm256_add_ps(finC, finD);
               __m256 Div = _mm256_load_ps(&divVec8[index]);

               __m256 finF = _mm256_add_ps(finE, Div);



               __m256 final = _mm256_mul_ps(finF, six6);



               _mm256_store_ps(&fromVec8[index], final);


            }
        }

        benchmark::DoNotOptimize(toVec8.data());
        benchmark::DoNotOptimize(fromVec8.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.

    }
    

}
// Register the function as a benchmark
BENCHMARK(pressurePaddedsimd8);



static void pressurePaddedsimd8Unaligned(benchmark::State& state) {


    //std::fill(begin(fromVec8), end(fromVec8), 1);

    auto fromVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    auto toVec8 = std::vector<float, boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    auto divVec8 = std::vector<float, boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    __m256 six6 = _mm256_set1_ps(0.16666666f);

    while (state.KeepRunning()){



        for (std::size_t k = 1; k < dimension8-1; ++k){

            for (std::size_t j = 1; j < dimension8-1; ++j){

               const std::size_t index = coord3to1Edge(j, k, dimension8);

               __m256 U = _mm256_load_ps(&fromVec8[index+dimension8]);
               __m256 D = _mm256_load_ps(&fromVec8[index-dimension8]);
               __m256 F = _mm256_load_ps(&fromVec8[index+dimension8Squared]);
               __m256 B = _mm256_load_ps(&fromVec8[index-dimension8Squared]);


               __m256 R = _mm256_loadu_ps(&fromVec8[index-1]);
               //p2 = a6, a5, a4, a3, a2, a1, a0, 0


               __m256 L = _mm256_loadu_ps(&fromVec8[index+1]);
               //p2 = 0, a7, a6, a5, a4, a3, a2, a1


               __m256 finA = _mm256_add_ps(L, R);
               __m256 finB = _mm256_add_ps(U, D);
               __m256 finC = _mm256_add_ps(F, B);

               __m256 finD = _mm256_add_ps(finA, finB);
               __m256 finE = _mm256_add_ps(finC, finD);

               __m256 Div = _mm256_load_ps(&divVec8[index]);

               __m256 finF = _mm256_add_ps(finE, Div);

               __m256 final = _mm256_mul_ps(finF, six6);

               _mm256_store_ps(&toVec8[index], final);


            }
        }
        for (std::size_t k = 2; k < dimension8-2; ++k){
            for (std::size_t j = 2; j < dimension8-2; ++j){

               const std::size_t index = coord3to1Edge(j, k, dimension8);

               __m256 U = _mm256_load_ps(&toVec8[index+dimension8]);
               __m256 D = _mm256_load_ps(&toVec8[index-dimension8]);
               __m256 F = _mm256_load_ps(&toVec8[index+dimension8Squared]);
               __m256 B = _mm256_load_ps(&toVec8[index-dimension8Squared]);

               __m256 R = _mm256_loadu_ps(&toVec8[index-1]);
               //p2 = a6, a5, a4, a3, a2, a1, a0, 0


               __m256 L = _mm256_loadu_ps(&toVec8[index+1]);
               //p2 = 0, a7, a6, a5, a4, a3, a2, a1


               __m256 finA = _mm256_add_ps(L, R);
               __m256 finB = _mm256_add_ps(U, D);
               __m256 finC = _mm256_add_ps(F, B);

               __m256 finD = _mm256_add_ps(finA, finB);
               __m256 finE = _mm256_add_ps(finC, finD);

               __m256 Div = _mm256_load_ps(&divVec8[index]);

               __m256 finF = _mm256_add_ps(finE, Div);


               __m256 final = _mm256_mul_ps(finF, six6);



               _mm256_store_ps(&fromVec8[index], final);


            }
        }

        benchmark::DoNotOptimize(toVec8.data());
        benchmark::DoNotOptimize(fromVec8.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.

    }


}
// Register the function as a benchmark
BENCHMARK(pressurePaddedsimd8Unaligned);

static void pressurePadded2d(benchmark::State& state) {

    auto fromVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);
    auto toVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);
    auto divVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);
    fromVec8.shrink_to_fit();
    toVec8.shrink_to_fit();
    divVec8.shrink_to_fit();
    for (auto& e : fromVec8){
        e = rand();
    }
    for (auto& e : toVec8){
        e = rand();
    }
    for (auto& e : divVec8){
        e = rand();
    }


    for (auto _ : state){
        for (int it = 0; it < 2; it++){
//#pragma omp parallel for
        for (std::size_t  j = 1; j < dimension512-1; ++j){
            for (std::size_t  i = 1; i < dimension512-1; ++i){

                       const  unsigned long  index = coord2dto1(i, j, dimension512);
                       //if (i == 0 || j == 0 || i == dimension512-1 || j == dimension512-1) continue;

                       toVec8[index] =  fromVec8[index-1] +
                                        fromVec8[index+1] +
                                        fromVec8[index-dimension512] +
                                        fromVec8[index+dimension512] +
                                        divVec8[index]/4.0f;

                       //count++;

            }
        }
//#pragma omp parallel for
        for (std::size_t  j = 1; j < dimension512-1; ++j){
            for (std::size_t  i = 1; i < dimension512-1; ++i){
                     const  unsigned long  index = coord2dto1(i, j, dimension512);

                       //if (i == 0 || j == 0 || i == dimension512-1 || j == dimension512-1) continue;
                       fromVec8[index] =  toVec8[index-1] +
                                        toVec8[index+1] +
                                        toVec8[index-dimension512] +
                                        toVec8[index+dimension512] +
                                        divVec8[index]/4.0f;
                       //count++;


            }
        }


        benchmark::DoNotOptimize(toVec8.data());

        benchmark::DoNotOptimize(fromVec8.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.
        }
        //std::cout<<count<<"\n";


    }

}
//BENCHMARK(pressurePadded2d);

static void pressurePadded2dTiled(benchmark::State& state) {

    auto fromVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);
    auto toVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);
    auto divVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);
    fromVec8.shrink_to_fit();
    toVec8.shrink_to_fit();
    divVec8.shrink_to_fit();
    for (auto& e : fromVec8){
        e = rand();
    }
    for (auto& e : toVec8){
        e = rand();
    }
    for (auto& e : divVec8){
        e = rand();
    }

    const std::size_t iblockSize = state.range(0);
    const std::size_t jblockSize = state.range(1);

    int count = 0;

    for (auto _ : state) {
        //count = 0;
        for (int it = 0; it < 2; it++){
            for (std::size_t jj = 0; jj < dimension512; jj+=jblockSize){
                for (std::size_t ii = 0; ii < dimension512; ii+=iblockSize){
                    for (std::size_t j = std::max(1ul,jj); j < std::min(jj+jblockSize,dimension512-1); ++j){
                        for (std::size_t i = std::max(1ul,ii); i < std::min(ii+iblockSize, dimension512-1); ++i){
//                    for (std::size_t j = jj; j < jj+jblockSize; ++j){
//                        for (std::size_t i = ii; i < ii+iblockSize; ++i){


                //if (i == 0 || j == 0 || i == dimension512-1 || j == dimension512-1) continue;
//                            i = std::max(1ul, i);
//                            i = std::min(dimension512-2ul, i);
//                            j = std::max(1ul, j);
//                            j = std::min(dimension512-2ul, j);

                        //const int index = coord2dto1(i, j, dimension512);
                        const unsigned long index = coord2dto1T<dimension512>(i, j);


                       toVec8[index] =  fromVec8[index-1] +
                                        fromVec8[index+1] +
                                        fromVec8[index-dimension512] +
                                        fromVec8[index+dimension512] +
                                        divVec8[index]/4.0f;
            }
        }
            }
        }

            for (std::size_t jj = jblockSize; jj < dimension512-jblockSize; jj+=jblockSize){
                for (std::size_t ii = iblockSize; ii < dimension512-iblockSize; ii+=iblockSize){
            for (std::size_t j = std::max(1ul,jj); j < std::min(jj+jblockSize,dimension512-1); ++j){
                for (std::size_t i = std::max(1ul,ii); i < std::min(ii+iblockSize, dimension512-1); ++i){
//                    for (std::size_t j = jj; j < jj+jblockSize; ++j){
//                        for (std::size_t i = ii; i < ii+iblockSize; ++i){


//                    int i = ii+_i;
//                    int j= jj+_j;

                      // if (i == 0 || j == 0 || i == dimension512-1 || j == dimension512-1) continue;
//                    i = std::max(1ul, i);
//                    i = std::min(dimension512-2ul, i);
//                    j = std::max(1ul, j);
//                    j = std::min(dimension512-2ul, j);

                    //const int index = coord2dto1(i, j, dimension512);
                    const unsigned long index = coord2dto1T<dimension512>(i, j);


                       fromVec8[index] = toVec8[index-1] +
                                        toVec8[index+1] +
                                        toVec8[index-dimension512] +
                                        toVec8[index+dimension512] +
                                        divVec8[index]/4.0f;
            }
        }
            }
        }


        benchmark::DoNotOptimize(toVec8.data());

        benchmark::DoNotOptimize(fromVec8.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.
        }

    }

}
//BENCHMARK(pressurePadded2dTiled)->Arg(4)->Arg(6)->Arg(8)->Arg(16)->Arg(32)->Arg(64)->Arg(128);
//BENCHMARK(pressurePadded2dTiled)->RangeMultiplier(2)->Ranges({{4, 64}, {4,64}});
//BENCHMARK(pressurePadded2dTiled)->Args({16, 16});
//BENCHMARK(pressurePadded2dTiled)->RangeMultiplier(2)->Range(4, 256}, {);

static void memoryLayoutA(benchmark::State& state) {

    auto image = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);

    for (auto _ : state) {
        for (std::size_t j = 0; j < dimension512; j++){
            for (std::size_t i = 0; i < dimension512; i++){

                const int index = coord2dto1(i, j, dimension512);

                image[index] =  i*i+j*j+i*j+i*j;

            }
        }
        benchmark::DoNotOptimize(image.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.
    }
}


//BENCHMARK(memoryLayoutA);

static void memoryLayoutB(benchmark::State& state) {

    auto image = std::vector<float,boost::alignment::aligned_allocator<float, 16>>(dimension512Squared);

    for (auto _ : state) {
        for (std::size_t j = 0; j < dimension512; j++){
            for (std::size_t i = 0; i < dimension512; i++){

                const int index = coord2dto1Inv(i, j, dimension512);

                image[index] =  i*i+j*j+i*j+i*j;

            }
        }
        benchmark::DoNotOptimize(image.data());
        benchmark::ClobberMemory(); // Force 42 to be written to memory.
        }
    }


//BENCHMARK(memoryLayoutB);

//8 padded to 12 - get 4 iterations done
static void pressurePadded12(benchmark::State& state) {
  while (state.KeepRunning()){
    //do stuff here
}
}
// Register the function as a benchmark
//BENCHMARK(pressurePadded12);

//BENCHMARK_MAIN();
int main (int argc,  char ** argv) {
    benchmark::Initialize(&argc, argv);
      benchmark::RunSpecifiedBenchmarks();
    return 0;
}
