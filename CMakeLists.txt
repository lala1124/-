cmake_minimum_required(VERSION 2.6)

project(fire)

add_definitions(-std=c++11)
add_definitions(-DAPI_EXPORTS)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

if(WIN32)
enable_language(CUDA)
endif(WIN32)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/home/nano/文档/fire/include)
link_directories(/home/nano/文档/fire/lib)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different


if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    message("embed_platform on")
    include_directories(/usr/local/cuda/targets/aarch64-linux/include)
    link_directories(/usr/local/cuda/targets/aarch64-linux/lib)
else()
    message("embed_platform off")
    # cuda
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)

    # tensorrt
    include_directories(/usr/include/aarch64-linux-gnu)
    link_directories(/usr/lib/aarch64-linux-gnu)
endif()


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -g -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
find_package(Threads)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(fire fire.cpp)

target_link_libraries(fire ${OpenCV_LIBS}
 /home/nano/文档/fire/lib/libonnxruntime.so
 /home/nano/文档/fire/lib/libonnxruntime_providers_cuda.so
 /home/nano/文档/fire/lib/libonnxruntime_providers_shared.so
 /home/nano/文档/fire/lib/libonnxruntime_providers_tensorrt.so
 ${CMAKE_THREAD_LIBS_INIT})


if(UNIX)
add_definitions(-O2 -pthread)
endif(UNIX)


