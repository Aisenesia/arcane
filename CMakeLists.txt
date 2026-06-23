cmake_minimum_required(VERSION 3.16)
project(arcane_dll LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package. vcpkg integration will resolve this automatically.
find_package(OpenCV REQUIRED)

# Check for CUDA (optional)
find_package(CUDA QUIET)

# Source files
set(SOURCES
    arcane-dll/arcane_dll.cpp
    arcane-dll/arcane_dll.h
    arcane-dll/dllmain.cpp
    arcane-dll/pch.cpp
    arcane-dll/pch.h
)

# Define shared library
add_library(arcane_dll SHARED ${SOURCES})

# Target compiler definitions
target_compile_definitions(arcane_dll PRIVATE ARCANE_DLL_EXPORTS)

# Include directories
target_include_directories(arcane_dll PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/arcane-dll
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(arcane_dll PRIVATE
    ${OpenCV_LIBS}
)

# Setup precompiled headers
target_precompile_headers(arcane_dll PRIVATE arcane-dll/pch.h)

# CUDA conditional support
if(CUDA_FOUND)
    target_compile_definitions(arcane_dll PRIVATE HAVE_CUDA)
    target_include_directories(arcane_dll PRIVATE ${CUDA_INCLUDE_DIRS})
    target_link_libraries(arcane_dll PRIVATE ${CUDA_LIBRARIES})
    message(STATUS "CUDA found, compiling with CUDA support.")
else()
    message(STATUS "CUDA not found, compiling CPU-only fallback.")
endif()
