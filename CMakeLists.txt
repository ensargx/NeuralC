cmake_minimum_required(VERSION 3.10)

project(neuralc)

set(CMAKE_C_STANDART 11)
set(CMAKE_C_CSTANDART_REQUIRED ON)

set(SRC_DIR neuralc)

file(GLOB_RECURSE SRC_FILES "${SRC_DIR}/*.c")

include_directories(${SRC_DIR})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_executable(neuralc ${SRC_FILES})

# Warnings and errors
target_compile_options(neuralc PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
)

# include the math library required for math.h
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    target_link_libraries(neuralc PUBLIC ${MATH_LIBRARY})
endif()

