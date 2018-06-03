cmake_minimum_required(VERSION 2.8.12)  

PROJECT(dlib_facedetector)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2 -DDLIB_JPEG_SUPPORT")

IF(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Weverything")
ELSEIF(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
ENDIF()

#INCLUDE OPENCV
FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
message(STATUS "Opencv include dir found at ${OpenCV_INCLUDE_DIRS}")

add_subdirectory(/home/nvidia/lzh/dlib-19.12/dlib dlib_build)

#包含头文件
INCLUDE_DIRECTORIES(/home/nvidia/lzh/dlib-19.12/) #dlib根目录地址

LINK_DIRECTORIES(/home/nvidia/lzh/dlib-19.12/build/dlib/) #dlib编译后bulid下dlib地址

#生产类似于.exe的可执行文件
ADD_EXECUTABLE(dlib_detector ${PROJECT_SOURCE_DIR}/face_detect.cpp)
#链接库
TARGET_LINK_LIBRARIES(dlib_detector dlib::dlib ${OpenCV_LIBS})
#TARGET_LINK_LIBRARIES(dlib_detector libjpeg.so)
