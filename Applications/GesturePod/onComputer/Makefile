# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.
 
APPNAME := gesturepodsim
CXX := g++
CXXFLAGS := -g -std=c++11
INCLUDES_PATH := -I./src/
SRC_PATH := ./src
OBJ_FILES := main.o featurizer.o protoNN.o utils.o

all: $(APPNAME)

$(APPNAME): $(OBJ_FILES)
	$(CXX) $(CXXFLAGS) $^ -o $@
	rm -f $(OBJ_FILES)

main.o: $(SRC_PATH)/main.cpp
	$(CXX) $(CXXFLAGS) -c $(INCLUDES_PATH) $< -o $@

featurizer.o : $(SRC_PATH)/featurizer.cpp $(SRC_PATH)/featurizer.h 
	$(CXX) $(CXXFLAGS) -c $(INCLUDES_PATH) $< -o $@

protoNN.o : $(SRC_PATH)/protoNN.cpp $(SRC_PATH)/protoNN.h 
	$(CXX) $(CXXFLAGS) -c $(INCLUDES_PATH) $< -o $@

utils.o : $(SRC_PATH)/utils.cpp $(SRC_PATH)/utils.h
	$(CXX) $(CXXFLAGS) -c $(INCLUDES_PATH) $< -o $@

clean:
	rm -f $(OBJ_FILES)
	rm -f $(APPNAME)