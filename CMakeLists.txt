cmake_minimum_required(VERSION 3.7)

set (CMAKE_CXX_STANDARD 17)

project ("vulkan_project")


find_package(Vulkan REQUIRED)
find_package(glfw3 3.4 REQUIRED)

add_executable (vulkan_project vulkan_test.cpp)


target_include_directories (vulkan_project 
    PUBLIC ${GLFW_INCLUDE_DIRS}
    PUBLIC ${VULKAN_INCLUDE_DIRS}
)
target_link_libraries (vulkan_project glfw)
target_link_libraries (vulkan_project Vulkan::Vulkan)