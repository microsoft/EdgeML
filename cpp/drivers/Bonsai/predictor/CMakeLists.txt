set (tool_name BonsaiPredict)

set (src BonsaiPredictDriver.cpp)

source_group("src" FILES ${src})

set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})

add_executable(${tool_name} ${src} ${include})
target_include_directories(${tool_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/common ${CMAKE_SOURCE_DIR}/src/Bonsai)

IF(CMAKE_COMPILER_IS_GNUCC)
 target_link_libraries(${tool_name} common Bonsai  mkl_intel_ilp64 mkl_core mkl_sequential cilkrts)
ENDIF (CMAKE_COMPILER_IS_GNUCC)

IF(NOT CMAKE_COMPILER_IS_GNUCC)
 target_link_libraries(${tool_name} common Bonsai  mkl_intel_ilp64 mkl_core mkl_sequential)
ENDIF (NOT CMAKE_COMPILER_IS_GNUCC)

set_property(TARGET ${tool_name} PROPERTY FOLDER "drivers/Bonsai")