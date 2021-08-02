cmake_minimum_required(VERSION 2.8)

project(YOLOv5-dnn)

find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_SOURCE_DIR}/inc)

file(GLOB  src  ${CMAKE_SOURCE_DIR}/src/*.cpp)
add_library(detect SHARED ${src})

add_executable(demo ${CMAKE_SOURCE_DIR}/demo/main.cpp)

target_link_libraries(demo 
	detect
	${OpenCV_LIBS}
	pthread 
	dl
	m)
