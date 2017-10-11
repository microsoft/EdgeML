set (library_name protonn)

set (src src/cluster.cpp
         src/ProtoNNPredictor.cpp             
         src/ProtoNNModel.cpp               
         src/ProtoNNTrainer.cpp
         src/ProtoNNFunctions.cpp    
         src/ProtoNN_Multiclass_export.cpp
         src/ProtoNNHyperParams.cpp  
         src/ProtoNNParams.cpp)

set (include include/cluster.h
             include/ProtoNNFunctions.h
             include/ProtoNN.h
             include/examples.h)

source_group("src" FILES ${src})
source_group("include" FILES ${include})

set(PARAMETER_SPARSITY_FLAGS -DSPARSE_LABEL) #-DSPARSE_Z #-DSPARSE_B #-DSPARSE_W 

set(PROTONN_FLAGS -DL2)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PARAMETER_SPARSITY_FLAGS} ${PROTONN_FLAGS}")

add_library(${library_name} ${src} ${include})

target_include_directories(${library_name} PUBLIC include ../../eigen)

target_link_libraries(${library_name} common)

set_property(TARGET ${library_name} PROPERTY FOLDER "protonn")
