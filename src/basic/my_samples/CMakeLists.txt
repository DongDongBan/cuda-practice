cmake_minimum_required(VERSION 3.12)
project(SAMPLES CUDA)

# set(CMAKE_CUDA_STANDARD 11)

file(GLOB SOURCE_FILES ${PROJECT_SOURCE_DIR}/*.cu)

foreach(srcfile  ${SOURCE_FILES})
    string( REGEX MATCH "[^/]+.cu$" program ${srcfile} )
    string( REPLACE ".cu" "" program ${program})
    add_executable( ${program} ${srcfile} )
    set_target_properties( ${program} PROPERTIES CUDA_ARCHITECTURES "75;86" )
endforeach(srcfile)


