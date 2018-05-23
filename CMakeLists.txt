cmake_minimum_required(VERSION 2.8)
find_package(CUDA REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

cuda_add_executable(
        test_knearests knearests.h knearests.cu kd_tree.cpp test_knearests.cu
        #OPTIONS "-arch sm_61 -lineinfo"
        #OPTIONS "-arch sm_30 -lineinfo"
)

