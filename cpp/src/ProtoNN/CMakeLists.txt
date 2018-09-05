set (library_name ProtoNN)

set (src cluster.h
         ProtoNNFunctions.h
         ProtoNN.h
         cluster.cpp
         ProtoNNModel.cpp               
         ProtoNNTrainer.cpp
         ProtoNNPredictor.cpp
         ProtoNNFunctions.cpp    
         ProtoNNHyperParams.cpp  
         ProtoNNParams.cpp)

source_group("src" FILES ${src})

set(PARAMETER_SPARSITY_FLAGS -DSPARSE_LABEL_PROTONN) #-DSPARSE_Z_PROTONN #-DSPARSE_B_PROTONN #-DSPARSE_W_PROTONN 

set(PROTONN_FLAGS -DL2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PARAMETER_SPARSITY_FLAGS} ${PROTONN_FLAGS}")

add_library(${library_name} ${src} ${include})

target_include_directories(${library_name} PUBLIC ../common ../../eigen)

target_link_libraries(${library_name} common)

set_property(TARGET ${library_name} PROPERTY FOLDER "ProtoNN")
