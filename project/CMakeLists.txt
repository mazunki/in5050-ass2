# CMakeLists.txt

cmake_minimum_required(VERSION 3.14)

#=== extracted from in5050-toolchain.cmake to make the life of griff easier :)
set(ENV{CUDA_HOME} "/usr/local/cuda")
set(ENV{PATH} "$ENV{CUDA_HOME}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")
include_directories(/opt/nvidia/nsight-systems/2023.2.4/target-linux-tegra-armv8/nvtx/include/nvtx3)
#===


#
# This tells CMake that your project's name is homeexam-01, and that the programs
# here are written in C.
#
project(Codec63
        VERSION 3.0
        DESCRIPTION "CMake configuration file for Codec63 in C"
        LANGUAGES CUDA C)

# Use the C standard from 2011. No effect on Windows (Visual Studio ignores
# the standards anyway).
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Overwrite default -O3 flag for Release mode
set(CMAKE_C_FLAGS_RELEASE "-O1 -fno-tree-vectorize")
# Make sure Debug mode flags are similar to Release mode flags

set(TRACE_LEVEL $ENV{TRACE_LEVEL})
if (NOT TRACE_LEVEL)
  set(TRACE_LEVEL 4)
endif()
set(CMAKE_C_FLAGS_DEBUG "-ggdb3 -pg -O1 -fno-tree-vectorize -DNDEBUG -DTRACE_LEVEL=${TRACE_LEVEL}")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -DNDEBUG -DTRACE_LEVEL=${TRACE_LEVEL}")

set(CMAKE_CUDA_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include)

# CMake configures Makefiles for discrete GPUs, but it doesn't do it
# for integrated GPUs like the Tegra. Pick the right compute capabilities
# explicitly. A bit hackish because a Tegra might have a discrete GPU.
# But it cannot use both at the same time.
if(EXISTS "/etc/nv_tegra_release")
  # The CC list for Tegras and Jetson will require manual updates
  set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")
else()
  # The CC list for discrete GPUs will require CMake updates
  set(CMAKE_CUDA_ARCHITECTURES "all-major"
      CACHE
      STRING "Which CUDA CCs to support: native, all, all-major or an explicit list delimited by semicolons")
endif()

# Debug compiles with -g or equivalent for debugging.
# Change to Release for good speed.
set(CMAKE_BUILD_TYPE $ENV{CMAKE_BUILD_TYPE})
if (NOT CMAKE_BUILD_TYPE)
  # set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_VERBOSE_MAKEFILE ON)

set(FP16_FLAGS "-march=armv8.2-a+fp+fp16+simd")

if(CMAKE_C_COMPILER_ID STREQUAL GNU)
  if(EXISTS "/etc/nv_tegra_release")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:${FP16_FLAGS}>)
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="${FP16_FLAGS}">)
  endif()
else()
  if(EXISTS "/etc/nv_tegra_release")
    add_compile_options($<$<COMPILE_LANGUAGE:C>:${FP16_FLAGS}>)
    # add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="${FP16_FLAGS}">)
  endif()
endif()

# This march option works with GCC but apparently not with the CUDA compiler on Tegra
# if(EXISTS "/etc/nv_tegra_release")
#     add_compile_options(-march=armv8.2-a)
# endif()

include_directories(include)

add_library(c63 STATIC
            src/quantdct.c src/quantdct.cu include/quantdct.h
            src/tables.cu include/tables.h
            src/io.cu include/io.h
            src/common.cu include/common.h
            src/me.cu include/me.h)
target_compile_options(c63 PUBLIC -I${CMAKE_SOURCE_DIR}/include)

add_executable(c63enc
               src/c63enc.cu include/c63.h
               src/c63_write.cu include/c63_write.h)
target_link_libraries(c63enc c63 m nvToolsExt)

add_executable(c63dec
               src/c63dec.cu include/c63.h)
target_link_libraries(c63dec c63 m nvToolsExt)

add_executable(c63pred
               src/c63dec.cu include/c63.h)
target_link_libraries(c63pred c63 m nvToolsExt)
target_compile_options(c63pred PUBLIC -DC63_PRED)
