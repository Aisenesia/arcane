cmake_minimum_required(VERSION 3.10.0)
set(CMAKE_BUILD_TYPE "")

project(dice_detect VERSION 0.1.0)

find_package(OpenCV REQUIRED)
# message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
# opencv found at
# message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

add_executable(arcane src/main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(arcane ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT})