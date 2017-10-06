set (tool_name bonsaitest)

set (src src/BonsaiLocalDriver.cpp)

source_group("src" FILES ${src})

set (GLOBAL_BIN_DIR ${CMAKE_BINARY_DIR}/bin)
set (EXECUTABLE_OUTPUT_PATH ${GLOBAL_BIN_DIR})
add_executable(${tool_name} ${src} ${include})
target_include_directories(${tool_name} PRIVATE include)

IF(CMAKE_COMPILER_IS_GNUCC)
 target_link_libraries(${tool_name} common bonsai mkl_sequential mkl_intel_ilp64 mkl_core gomp pthread cilkrts)
ENDIF (CMAKE_COMPILER_IS_GNUCC)

IF(NOT CMAKE_COMPILER_IS_GNUCC)
 target_link_libraries(${tool_name} common bonsai mkl_sequential mkl_intel_ilp64 mkl_core)
ENDIF (NOT CMAKE_COMPILER_IS_GNUCC)

set_property(TARGET ${tool_name} PROPERTY FOLDER "bonsaitest")
