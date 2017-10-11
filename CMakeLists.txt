cmake_minimum_required(VERSION 3.3)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(EdgeML)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Turn on ability to create folders to organize projects
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set global compilation flags
if(NOT WIN32 AND NOT CYGWIN)
  add_compile_options("-fvisibility-inlines-hidden")
endif()

# Set Visual Studio-specific options
if(MSVC)
  add_compile_options(/MP) #multi process build
endif()

#define variables for mkl include directories
set(MKL_INCLUDE_DIR $ENV{MKLROOT}/include)
include_directories(${MKL_INCLUDE_DIR})
link_directories($ENV{MKLROOT}/lib/intel64)

# add debug definitions to compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DLIGHT_LOGGER")  #-DLOGGER #-DTIMER -DCONCISE #-DSTDERR_ONSCREEN #-DLIGHT_LOGGER -DVERBOSE #-DDUMP #-DVERIFY

set(CONFIG_FLAGS "-DSINGLE") #-DXML -DZERO_BASED_IO

# mkl flags
set(MKL_EIGEN_FLAGS "-DEIGEN_USE_BLAS -DMKL_ILP64")

# add
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CONFIG_FLAGS} ${MKL_EIGEN_FLAGS}")

IF(CMAKE_COMPILER_IS_GNUCC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcilkplus -DCILK")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DLINUX")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -p -g")
ENDIF (CMAKE_COMPILER_IS_GNUCC)

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ox -DWINDOWS")
endif()

MESSAGE(STATUS "CMAKE_CXX_FLAGS:" ${CMAKE_CXX_FLAGS})
MESSAGE(STATUS "CMAKE_CXX_FLAGS_DEBUG:" ${CMAKE_CXX_FLAGS_DEBUG})

# Include project directories
add_subdirectory(lib)
add_subdirectory(test)
