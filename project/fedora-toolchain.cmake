# host-specific flags

set(CMAKE_C_COMPILER "/usr/bin/aarch64-linux-gnu-gcc" CACHE STRING "Use GCC for C compiler")
set(CMAKE_CXX_COMPILER "/usr/bin/aarch64-linux-gnu-g++" CACHE STRING "Use G++ for C++ compiler")


set(CUDA_HOME "/usr/local/cuda" CACHE STRING "CUDA Compiler")
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE STRING "CUDA Compiler")
set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/gcc-13" CACHE STRING "CUDA host compiler" FORCE)
set(CMAKE_CUDA_COMPILER_VERSION "11.4")

# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-march=armv8.2-a+fp16")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-mfpu=neon")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -ccbin=/usr/bin/gcc-13")

# Specify the sysroot path where the AArch64 headers are located
set(CMAKE_SYSROOT "/usr/aarch64-redhat-linux/sys-root/fc40")

# Ensure CMake searches for headers in the correct location
include_directories("/usr/aarch64-redhat-linux/sys-root/fc40/usr/include")


set(CMAKE_EXE_LINKER_FLAGS "-L/usr/local/cuda/lib64")

set(ENV{CUDA_HOME} "/usr/local/cuda")
set(ENV{PATH} "$ENV{CUDA_HOME}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")

include_directories("/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/compilers/include-stdpar/nvtx3")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
