set (library_name bonsai)

set (src src/Bonsai_Multiclass_export.cpp
         src/BonsaiHyperParams.cpp
         src/BonsaiModel.cpp
         src/BonsaiPredictor.cpp
         src/BonsaiFunctions.cpp
         src/BonsaiParams.cpp
         src/BonsaiTrainer.cpp)

set (include include/Bonsai.h
             include/BonsaiFunctions.h)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

set(PARAMETER_SPARSITY_FLAGS -DSPARSE_LABEL_BONSAI) #-DSPARSE_Z_BONSAI #-DSPARSE_V_BONSAI -DSPARSE_THETA_BONSAI -DSPARSE_W_BONSAI

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PARAMETER_SPARSITY_FLAGS}")

add_library(${library_name} ${src} ${include})

target_include_directories(${library_name} PUBLIC include ../../eigen)

target_link_libraries(${library_name} common)

set_property(TARGET ${library_name} PROPERTY FOLDER "bonsai")
