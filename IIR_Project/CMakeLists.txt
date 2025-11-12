cmake_minimum_required(VERSION 3.12)
project(iir_simd_bench CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(iir_bench iir_bench.cpp)

if (MSVC)
  target_compile_options(iir_bench PRIVATE /arch:AVX2)
else()
  target_compile_options(iir_bench PRIVATE -O3 -march=native -mavx2 -msse2 -mfma)
endif()
