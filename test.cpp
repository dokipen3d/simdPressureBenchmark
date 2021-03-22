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
constexpr int iterations = 1;
constexpr int slice = 1;


static void pressurePaddedSingle() {

    //fromVec8.clear();
    //std::fill(begin(fromVec8), end(fromVec8), 1);
    auto fromVec8 = std::vector<float>(dimension8*dimension8*dimension8);
    auto toVec8 = std::vector<float>(dimension8*dimension8*dimension8);
    auto divVec8 = std::vector<float>(dimension8*dimension8*dimension8);

    for (std::size_t k = 0; k < dimension8; ++k){
        for (std::size_t j = 0; j < dimension8; ++j){
            for (std::size_t i = 0; i < dimension8; ++i){
                const std::size_t index = coord3to1(i, j, k, dimension8);

                fromVec8[index] = index;
                divVec8[index] = index*37;
            }
        }
    }

    for (std::size_t k = 2; k < 3; ++k){
        std::cout << "\n";
        for (std::size_t j = 0; j < dimension8; ++j){
            std::cout << "\n";
            for (std::size_t i = 0; i < dimension8; ++i){
                const std::size_t index = coord3to1(i, j, k, dimension8);

                std::cout << fromVec8[index] << " ";
            }
        }
    }


    for(int l = 0; l < iterations;l++){
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

        for (std::size_t k = 2; k < 3; ++k){
            std::cout << "\n";
            for (std::size_t j = 0; j < dimension8; ++j){
                std::cout << "\n";
                for (std::size_t i = 0; i < dimension8; ++i){
                    const std::size_t index = coord3to1(i, j, k, dimension8);

                    std::cout << toVec8[index] << " ";
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
        for (std::size_t k = 2; k < 3; ++k){
            std::cout << "\n";
            for (std::size_t j = 0; j < dimension8; ++j){
                std::cout << "\n";
                for (std::size_t i = 0; i < dimension8; ++i){
                    const std::size_t index = coord3to1(i, j, k, dimension8);

                    std::cout << fromVec8[index] << " ";
                }
            }
        }
    }


    std::cout << "\n///////////////////////////////////////////////\n";
}

static void pressurePaddedsimd8() {



    auto fromVec8 = std::vector<float,boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    auto toVec8 = std::vector<float, boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);
    auto divVec8 = std::vector<float, boost::alignment::aligned_allocator<float, 32>>(dimension8*dimension8*dimension8);

    for (std::size_t k = 0; k < dimension8; ++k){
        for (std::size_t j = 0; j < dimension8; ++j){
            for (std::size_t i = 0; i < dimension8; ++i){
                const std::size_t index = coord3to1(i, j, k, dimension8);

                fromVec8[index] = index;
                divVec8[index] = index*37;
            }
        }
    }

    for (std::size_t k = 2; k < 3; ++k){
        std::cout << "\n";
        for (std::size_t j = 0; j < dimension8; ++j){
            std::cout << "\n";
            for (std::size_t i = 0; i < dimension8; ++i){
                const std::size_t index = coord3to1(i, j, k, dimension8);

                std::cout << fromVec8[index] << " ";
            }
        }
    }


    float(&fromArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(fromVec8.data());
    float(&toArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(toVec8.data());
    float(&divArray)[8][8][8]  = *reinterpret_cast<float(*)[8][8][8]>(divVec8.data());

    __m256 six6 = _mm256_set1_ps(0.166666666666666666666f);

    for(int l = 0; l < iterations;l++){

        for (std::size_t k = 1; k < dimension8-1; ++k){

            for (std::size_t j = 1; j < dimension8-1; ++j){

               const std::size_t index = coord3to1Edge(j, k, dimension8);

               __m256 U = _mm256_load_ps(&fromVec8[index+dimension8]);
               __m256 D = _mm256_load_ps(&fromVec8[index-dimension8]);
               __m256 F = _mm256_load_ps(&fromVec8[index+dimension8Squared]);
               __m256 B = _mm256_load_ps(&fromVec8[index-dimension8Squared]);

               //right shift
               __m256 R_a = _mm256_load_ps(&fromVec8[index]);
               __m256 p0 = _mm256_permute_ps(R_a, 0x93); //10010011=0x93
               __m256 p1 = _mm256_permute2f128_ps(p0, p0, 0x8); //00001000=0x8
               __m256 R = _mm256_blend_ps(p0, p1, 0x11); //p2

               //Left shift
               __m256 p0l = _mm256_permute_ps(R_a, 0x39); //00111001=0x39
               __m256 p1l = _mm256_permute2f128_ps(p0l, p0l, 0x81);//10000001=0x81
               __m256 L = _mm256_blend_ps(p0l, p1l, 0x88); //10001000=0x88 p2


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

        for (std::size_t k = 2; k < 3; ++k){
            std::cout << "\n";
            for (std::size_t j = 0; j < dimension8; ++j){
                std::cout << "\n";
                for (std::size_t i = 0; i < dimension8; ++i){
                    const std::size_t index = coord3to1(i, j, k, dimension8);

                    std::cout << toVec8[index] << " ";
                }
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
               __m256 p0 = _mm256_permute_ps(R_a, 0x93); //10010011=0x93
               __m256 p1 = _mm256_permute2f128_ps(p0, p0, 0x8); //00001000=0x8
               __m256 R = _mm256_blend_ps(p0, p1, 0x11); //p2

               //Left shift
               __m256 p0l = _mm256_permute_ps(R_a, 0x39); //00111001=0x39
               __m256 p1l = _mm256_permute2f128_ps(p0l, p0l, 0x81);//10000001=0x81
               __m256 L = _mm256_blend_ps(p0l, p1l, 0x88); //10001000=0x88 p2

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
        for (std::size_t k = 2; k < 3; ++k){
            std::cout << "\n";
            for (std::size_t j = 0; j < dimension8; ++j){
                std::cout << "\n";
                for (std::size_t i = 0; i < dimension8; ++i){
                    const std::size_t index = coord3to1(i, j, k, dimension8);

                    std::cout << fromVec8[index] << " ";
                }
            }
        }
    }



}

int main(){

    pressurePaddedSingle() ;
    pressurePaddedsimd8();
 
}
