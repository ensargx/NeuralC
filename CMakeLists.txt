cmake_minimum_required(VERSION 3.10)

project(neuralc)

set(CMAKE_C_STANDART 11)
set(CMAKE_C_CSTANDART_REQUIRED ON)

set(SRC_DIR neuralc)

file(GLOB_RECURSE SRC_FILES "${SRC_DIR}/*.c")

include_directories(${SRC_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(neuralc ${SRC_FILES})

