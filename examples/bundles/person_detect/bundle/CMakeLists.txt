cmake_minimum_required(VERSION 3.16)

project(visual_wake_words)

set(model "person_detect_quant")

# Executable.
add_executable(${model} $<TARGET_OBJECTS:BundleTFLiteCustomMain>)
target_link_libraries(${model} 
  ${CMAKE_CURRENT_LIST_DIR}/${model}.o
  -lpng 
  -lm 
  -ldl
  -no-pie
  -fPIC
  )
add_dependencies(${model} BundleTFLiteCustomMain BundleTFLiteCustomNet)
add_custom_target(BundleTFLiteCustomNet DEPENDS ${CMAKE_CURRENT_LIST_DIR}/${model}.o)

# Driver program with main function for regular bundle
add_library(BundleTFLiteCustomMain OBJECT main.cpp)
target_compile_options(BundleTFLiteCustomMain PRIVATE -std=c++11 -g )
add_dependencies(BundleTFLiteCustomMain BundleTFLiteCustomNet)
