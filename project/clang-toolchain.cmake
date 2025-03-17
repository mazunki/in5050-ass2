
set(CMAKE_C_COMPILER "/usr/bin/clang" CACHE STRING "Use Clang for C compiler")
set(CMAKE_CXX_COMPILER "/usr/bin/clang++" CACHE STRING "Use Clang++ for C++ compiler")
set(CMAKE_CUDA_COMPILER "/usr/bin/clang++" CACHE STRING "Use Clang++ for CUDA Compiler")

# set(CMAKE_CUDA_HOST_COMPILER "/usr/bin/clang++" CACHE STRING "Use Clang++ for CUDA Host Compiler" FORCE)

set(CUDA_HOME "/usr/local/cuda-11.4" CACHE STRING "CUDA Compiler")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --cuda-gpu-arch=sm_75 --cuda-path=${CUDA_HOME}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -march=armv8.2-a+simd")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -x cuda")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lcudart_static -ldl -lrt -pthread")

set(CMAKE_EXE_LINKER_FLAGS "-L${CUDA_HOME}/lib64 -lcudart_static -ldl -lrt -pthread")
set(CMAKE_EXE_LINKER_FLAGS "-L${CUDA_HOME}/lib64")

set(CMAKE_CUDA_ARCHITECTURES "53;62;72;87")

set(ENV{CUDA_HOME} "${CUDA_HOME}")
set(ENV{PATH} "$ENV{CUDA_HOME}/bin:$ENV{PATH}")
set(ENV{LD_LIBRARY_PATH} "$ENV{CUDA_HOME}/lib64:$ENV{LD_LIBRARY_PATH}")

