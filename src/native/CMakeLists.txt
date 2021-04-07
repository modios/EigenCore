# CMakeList.txt : Top-level CMake project file, do global configuration

cmake_minimum_required (VERSION 3.8)

project ("EigenNative")

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)

set(EIGEN_DIRECTORY . CACHE STRING "Eigen directory.  The directory specified should have a subdirectory 'Eigen'.")
include_directories(${EIGEN_DIRECTORY})

add_library(eigen_core SHARED EigenNative.cpp)