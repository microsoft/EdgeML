set (library_name Bonsai)

set (src Bonsai.h
         BonsaiFunctions.h  
         BonsaiHyperParams.cpp
         BonsaiModel.cpp
         BonsaiIngestTest.cpp
         BonsaiPredictor.cpp
         BonsaiFunctions.cpp
         BonsaiParams.cpp
         BonsaiTrainer.cpp)

source_group("src" FILES ${src})

set(PARAMETER_SPARSITY_FLAGS -DSPARSE_LABEL_BONSAI) #-DSPARSE_Z_BONSAI #-DSPARSE_V_BONSAI -DSPARSE_THETA_BONSAI -DSPARSE_W_BONSAI

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PARAMETER_SPARSITY_FLAGS}")

add_library(${library_name} ${src})

target_include_directories(${library_name} PUBLIC ../common ../../eigen)

target_link_libraries(${library_name} common)

set_property(TARGET ${library_name} PROPERTY FOLDER "Bonsai")
