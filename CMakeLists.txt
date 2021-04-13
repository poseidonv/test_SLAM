cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 14)

project(SLAM_base)

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(${PROJECT_NAME} SHARED
src/Converter.cc
src/Reprojection.cc
src/SLAM_base.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
)

add_executable(test test.cc)

target_link_libraries(test ${PROJECT_NAME})