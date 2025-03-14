cmake_minimum_required(VERSION 3.14)

project(Codec63
  VERSION 3.0
  DESCRIPTION "CMake configuration file for Codec63 in CUDA"
  LANGUAGES CUDA C)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
set(CMAKE_C_FLAGS_DEBUG "-ggdb3 -pg -G -O1 -fno-tree-vectorize -DNDEBUG")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -pg -pg")

set(CMAKE_CUDA_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_compile_options(-gencode arch=compute_72,code=sm_72 -Wno-deprecated-gpu-targets)

# Library for shared functionality
add_library(c63 STATIC
    src/quantdct.cu include/quantdct.h
    src/tables.cu include/tables.h
    src/io.cu include/io.h
    src/common.cu include/common.h
    src/me.cu include/me.h
)
target_compile_options(c63 PUBLIC -I${CMAKE_SOURCE_DIR}/include)

# Encoder executable
add_executable(c63enc
    src/c63enc.cu include/c63.h
    src/c63_write.cu include/c63_write.h
)
target_link_libraries(c63enc c63 m)

# Decoder executable
add_executable(c63dec
	src/c63dec.cu include/c63.h
)
target_link_libraries(c63dec c63 m)

# Prediction executable
add_executable(c63pred
	src/c63dec.cu include/c63.h
)
target_link_libraries(c63pred c63 m)
target_compile_options(c63pred PUBLIC -DC63_PRED)


if(EXISTS "/etc/nv_tegra_release")
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
    target_compile_options(c63 PRIVATE -Xcompiler=-march=armv8.2-a)
    target_compile_options(c63enc PRIVATE -Xcompiler=-march=armv8.2-a)
    target_compile_options(c63dec PRIVATE -Xcompiler=-march=armv8.2-a)
    target_compile_options(c63pred PRIVATE -Xcompiler=-march=armv8.2-a)
  endif()
endif()

