// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"

using namespace std;

enum Algo { Bonsai, Protonn };
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

	if ((int)features.size() != featuresLength)
		throw "Number of row entries in X is inconsistent";

	return features;
}

int getLabel(string line) {
	static int labelLength = -1;

	vector<string> labels = readCSVLine(line);

	if (labelLength == -1)
		labelLength = (int)labels.size();

	if ((int)labels.size() != labelLength || labels.size() != 1)
		throw "Number of row entries in Y is inconsistent";

	return (int)atoi(labels.front().c_str());
}

int main(int argc, char *argv[]) {
	if (argc == 1) {
		cout << "No arguments supplied" << endl;
		return 1;
	}

	// Parsing the arguments
	Algo algo;
	if (strcmp(argv[1], "bonsai") == 0)
		algo = Bonsai;
	else if (strcmp(argv[1], "protonn") == 0)
		algo = Protonn;
	else {
		cout << "Argument mismatch for algo\n";
		return 1;
	}
	string algoStr = argv[1];

	Version version;
	if (strcmp(argv[2], "fixed") == 0)
		version = Fixed;
	else if (strcmp(argv[2], "float") == 0)
		version = Float;
	else {
		cout << "Argument mismatch for version\n";
		return 1;
	}
	string versionStr = argv[2];

	DatasetType datasetType;
	if (strcmp(argv[3], "training") == 0)
		datasetType = Training;
	else if (strcmp(argv[3], "testing") == 0)
		datasetType = Testing;
	else {
		cout << "Argument mismatch for dataset type\n";
		return 1;
	}
	string datasetTypeStr = argv[3];

	// Reading the dataset
	string inputDir = "input/";

	ifstream featuresFile(inputDir + "X.csv");
	ifstream lablesFile(inputDir + "Y.csv");

	if (featuresFile.good() == false || lablesFile.good() == false)
		throw "Input files doesn't exist";

	// Create output directory and files
	string outputDir = "output/" + algoStr + "-" + versionStr;

	string outputFile = outputDir + "/prediction-info-" + datasetTypeStr + ".txt";
	string statsFile = outputDir + "/stats-" + datasetTypeStr + ".txt";

	ofstream output(outputFile);
	ofstream stats(statsFile);

	int correct = 0, total = 0;

	bool alloc = false;
	int features_size = -1;
	MYINT **features_int = NULL;
	float *features_float = NULL;

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

			if (version == Fixed) {
				features_int = new MYINT*[features_size];
				for (int i = 0; i < features_size; i++)
					features_int[i] = new MYINT[1];
			}
			else
				features_float = new float[features_size];

			alloc = true;
		}

		// Populate the array using the feature vector
		if (version == Fixed)
			for (int i = 0; i < features_size; i++) {
#ifdef INT16
				features_int[i][0] = (MYINT)(atol(features.at(i).c_str()));
#endif
#ifdef INT32
				features_int[i][0] = (MYINT)(atoll(features.at(i).c_str()));
#endif
			}
		else
			for (int i = 0; i < features_size; i++)
				features_float[i] = (float)(atof(features.at(i).c_str()));

		// Invoke the predictor function
		int res = -1;
		if (algo == Bonsai && version == Fixed)
			res = seedotFixed(features_int);
		else if (algo == Bonsai && version == Float)
			res = bonsaiFloat(features_float);
		else if (algo == Protonn && version == Fixed)
			res = seedotFixed(features_int);
		else if (algo == Protonn && version == Float)
			res = protonnFloat(features_float);

		if ((res) == label) {
			correct++;
		}
		else {
			output << "Incorrect prediction for input " << total + 1 << ". Predicted " << res + 1 << " Expected " << label << endl;
		}

		total++;
	}

	// Deallocate memory
	if (version == Fixed) {
		for (int i = 0; i < features_size; i++)
			delete features_int[i];
		delete features_int;
	}
	else
		delete features_float;

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
		dumpRange(outputDir + "/profile.txt");

	return 0;
}
