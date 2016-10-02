project(Flower)
cmake_minimum_required(VERSION 2.8)
aux_source_directory(. SRC_LIST)
aux_source_directory(src/flower SRC_LIST)
aux_source_directory(src/flower/layer SRC_LIST)
aux_source_directory(src/flower/optimizer SRC_LIST)
add_executable(${PROJECT_NAME} ${SRC_LIST})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -DEIGEN_MPL2_ONLY")

# add header
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen3)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/eigen3/unsupported)

# add src
add_subdirectory(src/flower)
add_subdirectory(src/flower/layer)

# add header to qt project folder
FILE(GLOB_RECURSE LibFiles "include/*.h" "include/*.inl")
add_custom_target(headers SOURCES ${LibFiles})
