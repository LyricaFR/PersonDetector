cmake_minimum_required(VERSION 2.8)
project( PersonDetector )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( PersonDetector ./src/PersonDetector.cpp )

find_package(OpenMP)

target_link_libraries( PersonDetector ${OpenCV_LIBS} OpenMP::OpenMP_CXX)
set (CMAKE_CXX_FLAGS "-lstdc++fs -std=c++17")