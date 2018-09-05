set (library_name common)

set (src blas_routines.h
         Data.h
         goldfoil.h
         logger.h
         mmaped.h
         metrics.h
         par_utils.h
         pre_processor.h
         timer.h
         utils.h
         blas_routines.cpp
         Data.cpp
         goldfoil.cpp
         logger.cpp
         mmaped.cpp
         metrics.cpp
         par_utils.cpp
         timer.cpp
         utils.cpp)


source_group("src" FILES ${src})

add_library(${library_name} ${src})

target_include_directories(${library_name} PUBLIC ../../eigen)

set_property(TARGET ${library_name} PROPERTY FOLDER "common")
