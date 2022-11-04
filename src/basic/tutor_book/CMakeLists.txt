cmake_minimum_required(VERSION 3.12)
project(REDUCE CUDA)

set(CMAKE_CUDA_STANDARD 17)

file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cu)
add_executable(reduce ${SOURCE_FILES})

set_target_properties(reduce PROPERTIES CUDA_ARCHITECTURES "75;86")

if(USE_DOUBLE)
    target_compile_definitions(reduce PRIVATE USE_DOUBLE)
endif()
