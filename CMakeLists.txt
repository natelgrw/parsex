cmake_minimum_required(VERSION 3.20)
project(parsex_compiler CXX)

# C++ Standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Helper for macOS OpenMP (Homebrew)
if(APPLE)
    if(EXISTS "/opt/homebrew/opt/libomp")
        set(OpenMP_ROOT "/opt/homebrew/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
    elseif(EXISTS "/usr/local/opt/libomp")
        set(OpenMP_ROOT "/usr/local/opt/libomp")
        list(APPEND CMAKE_PREFIX_PATH "/usr/local/opt/libomp")
    endif()
endif()

# Find OpenMP
find_package(OpenMP REQUIRED)

# Find Eigen3
# Trying standard find first (should be found via Homebrew now)
find_package(Eigen3 REQUIRED NO_MODULE)


# Include directories
include_directories(include)
include_directories(include/cpu)
include_directories(include/gpu)
include_directories(include/parallel)

# Sources
set(SOURCES
    src/MNASolver.cpp
    src/cpu/CpuSolver.cpp
)

# CUDA Support
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND SOURCES src/gpu/GpuSolver.cu)
    set(CUDA_ENABLED ON)
else()
    message(STATUS "CUDA not found. GPU module will be disabled.")
    add_definitions(-DNO_CUDA)
    set(CUDA_ENABLED OFF)
endif()

# Batch Solver Executable
add_executable(parsex_batch src/batch_main.cpp ${SOURCES})

if(CUDA_ENABLED)
    target_link_libraries(parsex_batch
        PRIVATE
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
        CUDA::cublas
        CUDA::cudart
    )
else()
    target_link_libraries(parsex_batch
        PRIVATE
        OpenMP::OpenMP_CXX
        Eigen3::Eigen
    )
endif()

# Reference Serial Solver (Optional, keeping for regression testing)
add_executable(circuit_solver src/main.cpp src/MNASolver.cpp)
target_link_libraries(circuit_solver
    PRIVATE
    Eigen3::Eigen 
    # Serial solver doesn't strictly need OpenMP but good to have consistency
    OpenMP::OpenMP_CXX 
)
