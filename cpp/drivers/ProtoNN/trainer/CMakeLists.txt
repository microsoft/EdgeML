set (tool_name ProtoNNTrain)

set (src ProtoNNTrainDriver.cpp)

source_group("src" FILES ${src})

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})

add_executable(${tool_name} ${src} ${include})
target_include_directories(${tool_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/common ${CMAKE_SOURCE_DIR}/src/ProtoNN)

IF(CMAKE_COMPILER_IS_GNUCC)
 target_link_libraries(${tool_name} common ProtoNN mkl_intel_ilp64 mkl_core mkl_gnu_thread gomp pthread cilkrts)
ENDIF (CMAKE_COMPILER_IS_GNUCC)

IF(NOT CMAKE_COMPILER_IS_GNUCC)
 target_link_libraries(${tool_name} common ProtoNN mkl_intel_ilp64  mkl_intel_thread mkl_core libiomp5md)
ENDIF (NOT CMAKE_COMPILER_IS_GNUCC)

set_property(TARGET ${tool_name} PROPERTY FOLDER "drivers/ProtoNN")
