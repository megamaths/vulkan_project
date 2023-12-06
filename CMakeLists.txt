cmake_minimum_required(VERSION 3.7)

project(rayTracer)
set(CMAKE_CXX_STANDARD 17)

add_executable(rayTracer vulkan_test.cpp)
find_package(glfw3 3.0 REQUIRED)
target_link_libraries(rayTracer glfw)

find_package(Vulkan REQUIRED FATAL_ERROR)
target_link_libraries(rayTracer ${Vulkan_LIBRARIES})