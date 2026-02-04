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

# Sources
set(SOURCES
    src/MNASolver.cpp
    src/BatchSolver.cpp
)

# Batch Solver Executable
add_executable(parsex_batch src/batch_main.cpp ${SOURCES})
target_link_libraries(parsex_batch
    PRIVATE
    OpenMP::OpenMP_CXX
    Eigen3::Eigen
)

# Reference Serial Solver (Optional, keeping for regression testing)
add_executable(circuit_solver src/main.cpp src/MNASolver.cpp)
target_link_libraries(circuit_solver
    PRIVATE
    Eigen3::Eigen 
    # Serial solver doesn't strictly need OpenMP but good to have consistency
    OpenMP::OpenMP_CXX 
)
