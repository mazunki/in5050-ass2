# in5050-toolchain.cmake

set(CMAKE_C_COMPILER "/usr/bin/gcc" CACHE STRING "Use GCC for C compiler")
set(CMAKE_CXX_COMPILER "/usr/bin/g++" CACHE STRING "Use G++ for C++ compiler")

set(CUDA_HOME "/usr/local/cuda-11.4" CACHE STRING "CUDA Compiler")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.4/bin/nvcc" CACHE STRING "CUDA Compiler")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc" CACHE STRING "CUDA host compiler" FORCE)
set(CMAKE_CUDA_COMPILER_VERSION "11.4")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-march=armv8.2-a+simd+fp16")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/usr/bin/gcc")


set(CMAKE_EXE_LINKER_FLAGS "-L/usr/local/cuda-11.4/lib64")

set(ENV{CUDA_HOME} "/usr/local/cuda-11.4")
set(ENV{PATH} "$ENV{CUDA_HOME}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")

include_directories(/opt/nvidia/nsight-systems/2023.2.4/target-linux-tegra-armv8/nvtx/include/nvtx3)
