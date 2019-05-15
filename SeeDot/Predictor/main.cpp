// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"

using namespace std;

enum Version { Fixed, Float };
enum DatasetType { Training, Testing };

// Split the CSV row into multiple values
vector<string> readCSVLine(string line) {
	vector<string> tokens;

	stringstream stream(line);
	string str;

	while (getline(stream, str, ','))
		tokens.push_back(str);

	return tokens;
}

vector<string> getFeatures(string line) {
	static int featuresLength = -1;

	vector<string> features = readCSVLine(line);

	if (featuresLength == -1)
		featuresLength = (int)features.size();

	if (features.size() != featuresLength)
		throw "Number of row entries in X is inconsistent";

	return features;
}

int getLabel(string line) {
	static int labelLength = -1;

	vector<string> labels = readCSVLine(line);

	if (labelLength == -1)
		labelLength = (int)labels.size();

	if (labels.size() != labelLength || labels.size() != 1)
		throw "Number of row entries in Y is inconsistent";

	return (int)atoi(labels.front().c_str());
}

void populateFixedVector(MYINT** features_int, vector<string> features, int scale) {
	int features_size = (int)features.size();

	for (int i = 0; i < features_size; i++) {
		double f = (double)(atof(features.at(i).c_str()));
		double f_int = ldexp(f, -scale);
		features_int[i][0] = (MYINT)(f_int);
	}

	return;
}

void populateFloatVector(float **features_float, vector<string> features) {
	int features_size = (int)features.size();
	for (int i = 0; i < features_size; i++)
		features_float[i][0] = (float)(atof(features.at(i).c_str()));
	return;
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		cout << "No arguments supplied" << endl;
		return 1;
	}

	Version version;
	if (strcmp(argv[1], "fixed") == 0)
		version = Fixed;
	else if (strcmp(argv[1], "float") == 0)
		version = Float;
	else {
		cout << "Argument mismatch for version\n";
		return 1;
	}
	string versionStr = argv[1];

	DatasetType datasetType;
	if (strcmp(argv[2], "training") == 0)
		datasetType = Training;
	else if (strcmp(argv[2], "testing") == 0)
		datasetType = Testing;
	else {
		cout << "Argument mismatch for dataset type\n";
		return 1;
	}
	string datasetTypeStr = argv[2];

	// Reading the dataset
	string inputDir = "input\\";

	ifstream featuresFile(inputDir + "X.csv");
	ifstream lablesFile(inputDir + "Y.csv");

	if (featuresFile.good() == false || lablesFile.good() == false)
		throw "Input files doesn't exist";

	// Create output directory and files
	string outputDir = "output\\" + versionStr;

	string outputFile = outputDir + "\\prediction-info-" + datasetTypeStr + ".txt";
	string statsFile = outputDir + "\\stats-" + datasetTypeStr + ".txt";

	ofstream output(outputFile);
	ofstream stats(statsFile);

	int correct = 0, total = 0;

	bool alloc = false;
	int features_size = -1;
	MYINT **features_int = NULL;
	float **features_float = NULL;

	// Initialize variables used for profiling
	initializeProfiling();

	string line1, line2;
	while (getline(featuresFile, line1) && getline(lablesFile, line2)) {
		// Read the feature vector and class ID
		vector<string> features = getFeatures(line1);
		int label = getLabel(line2);

		// Allocate memory to store the feature vector as arrays
		if (alloc == false) {
			features_size = (int)features.size();

			features_int = new MYINT*[features_size];
			for (int i = 0; i < features_size; i++)
				features_int[i] = new MYINT[1];

			features_float = new float*[features_size];
			for (int i = 0; i < features_size; i++)
				features_float[i] = new float[1];

			alloc = true;
		}

		// Populate the array using the feature vector
		if (debugMode) {
			populateFixedVector(features_int, features, scaleForX);
			populateFloatVector(features_float, features);
		}
		else if (version == Fixed)
			populateFixedVector(features_int, features, scaleForX);
		else
			populateFloatVector(features_float, features);

		// Invoke the predictor function
		int res;
		
		if (debugMode) {
			int res_float = seedotFloat(features_float);
			int res_fixed = seedotFixed(features_int);
			//debug();
			res = res_fixed;
		}
		else {
			if (version == Fixed)
				res = seedotFixed(features_int);
			else if (version == Float)
				res = seedotFloat(features_float);
		}

		if ((res + 1) == label) {
			correct++;
		}
		else {
			output << "Incorrect prediction for input " << total + 1 << ". Predicted " << res + 1 << " Expected " << label << endl;
		}

		total++;
	}

	// Deallocate memory
	for (int i = 0; i < features_size; i++)
		delete features_int[i];
	delete[] features_int;
	
	for (int i = 0; i < features_size; i++)
		delete features_float[i];
	delete[] features_float;

	float accuracy = (float)correct / total * 100.0f;

	cout.precision(3);
	cout << fixed;
	cout << "\n\n#test points = " << total << endl;
	cout << "Correct predictions = " << correct << endl;
	cout << "Accuracy = " << accuracy << "\n\n";

	output.precision(3);
	output << fixed;
	output << "\n\n#test points = " << total << endl;
	output << "Correct predictions = " << correct << endl;
	output << "Accuracy = " << accuracy << "\n\n";
	output.close();

	stats.precision(3);
	stats << fixed;
	stats << accuracy << "\n";
	stats.close();

	if (datasetType == Training)
		dumpRange(outputDir + "\\profile.txt");

	return 0;
}
