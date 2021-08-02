cmake_minimum_required(VERSION 2.8)

project(YOLOv5-dnn)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})

file(GLOB  src  ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_executable(demo ${src})

target_link_libraries(demo ${OpenCV_LIBS}
	pthread 
	dl
	m)
