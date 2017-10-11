set (library_name common)

set (src src/blas_routines.cpp
         src/Data.cpp
         src/goldfoil.cpp
         src/logger.cpp
         src/mmaped.cpp
         src/par_utils.cpp
         src/timer.cpp
         src/utils.cpp)

set (include include/blas_routines.h
             include/Data.h
             include/goldfoil.h
             include/logger.h
             include/mmaped.h
             include/par_utils.h
             include/pre_processor.h
             include/timer.h
             include/utils.h)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

add_library(${library_name} ${src} ${include})

target_include_directories(${library_name} PUBLIC include ../../eigen)

set_property(TARGET ${library_name} PROPERTY FOLDER "common")
