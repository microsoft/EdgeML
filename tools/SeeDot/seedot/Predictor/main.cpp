// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <list>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <thread>
#include <algorithm>

#include "datatypes.h"
#include "predictors.h"
#include "profile.h"

using namespace std;

/*
 * This file is the driver for the x86 version of the code. It reads the floating point data from the csv files, parses them
 * and translates them into integers, and then puts them through multiple generated inference codes and evaluates each result.
 */

enum Version
{
	Fixed,
	Float
};
enum DatasetType
{
	Training,
	Testing
};
enum ProblemType
{
	Classification,
	Regression
};

bool profilingEnabled = false;

// Split the CSV row into multiple values.
vector<string> readCSVLine(string line) {
	vector<string> tokens;

	stringstream stream(line);
	string str;

	while (getline(stream, str, ',')) {
		tokens.push_back(str);
	}

	return tokens;
}

// Read the input 'X'.
vector<string> getFeatures(string line) {
	static int featuresLength = -1;

	vector<string> features = readCSVLine(line);

	if (featuresLength == -1) {
		featuresLength = (int)features.size();
	}

	if ((int)features.size() != featuresLength) {
		throw "Number of row entries in X is inconsistent";
	}

	return features;
}

// Read the ground truth label/value 'Y'.
vector<string> getLabel(string line) {
	static int labelLength = -1;

	vector<string> labels = readCSVLine(line);

	if (labelLength == -1) {
		labelLength = (int)labels.size();
	}

	if ((int)labels.size() != labelLength) {
		throw "Number of row entries in Y is inconsistent";
	}

	return labels;
}

// Take in the input floating point datapoint, convert it to a fixed point integer and store it.
void populateFixedVector(MYINT** features_int, vector<string> features, int scale) {
	int features_size = (int)features.size();

	for (int i = 0; i < features_size; i++) {
		double f = (double)(atof(features.at(i).c_str()));
		double f_int = ldexp(f, -scale);
		features_int[i][0] = (MYINT)(f_int);
	}

	return;
}

// Take in the input floating point datapoint and store it.
void populateFloatVector(float** features_float, vector<string> features) {
	int features_size = (int)features.size();
	for (int i = 0; i < features_size; i++) {
		features_float[i][0] = (float)(atof(features.at(i).c_str()));
	}
	return;
}

// Multi-threading is used to speed up exploration.
// Each thread, which invokes the following method, is responsible for taking in one datapoint
// and running it through all the generated codes.
// Number of threads generated equals the number of datapoints in the given dataset.
void launchThread(int features_size, MYINT** features_int, MYINT*** features_intV, float** features_float, int counter, float* float_res, int* res, int** resV) {
	seedotFixed(features_int, res);
	seedotFloat(features_float, float_res);

	for (int i = 0; i < switches; i++) {
		seedotFixedSwitch(i, features_intV[i], resV[i]);
	}

	for (int i = 0; i < features_size; i++) {
		delete features_int[i];
		delete features_float[i];
		for (int j = 0; j < switches; j++) {
			delete features_intV[j][i];
		}
	}
	delete[] features_int;
	delete[] features_float;
	for (int j = 0; j < switches; j++) {
		delete[] features_intV[j];
	}
	delete[] features_intV;
}

int main(int argc, char* argv[]) {
	float epsilon = 0.00001;
	if (argc == 1) {
		cout << "No arguments supplied" << endl;
		return 1;
	}

	Version version;
	if (strcmp(argv[1], "fixed") == 0) {
		version = Fixed;
	} else if (strcmp(argv[1], "float") == 0) {
		version = Float;
	} else {
		cout << "Argument mismatch for version\n";
		return 1;
	}
	string versionStr = argv[1];

	DatasetType datasetType;
	if (strcmp(argv[2], "training") == 0) {
		datasetType = Training;
	} else if (strcmp(argv[2], "testing") == 0) {
		datasetType = Testing;
	} else {
		cout << "Argument mismatch for dataset type\n";
		return 1;
	}
	string datasetTypeStr = argv[2];

	ProblemType problem;
	if (strcmp(argv[3], "classification") == 0) {
		problem = Classification;
	} else if (strcmp(argv[3], "regression") == 0) {
		problem = Regression;
	} else {
		cout << "Argument mismatch for problem type\n";
		return 1;
	}
	string problemTypeStr = argv[3];

	int numOutputs = atoi(argv[4]);

	// Reading the dataset.
	string inputDir = "input/";

	ifstream featuresFile(inputDir + "X.csv");
	ifstream lablesFile(inputDir + "Y.csv");

	if (featuresFile.good() == false || lablesFile.good() == false) {
		throw "Input files doesn't exist";
	}

	// Create output directory and files.
	string outputDir = "output/" + versionStr;

	string outputFile = outputDir + "/prediction-info-" + datasetTypeStr + ".txt";
	string statsFile = outputDir + "/stats-" + datasetTypeStr + ".txt";

	ofstream output(outputFile);
	ofstream stats(statsFile);

	bool alloc = false;
	int features_size = -1;
	MYINT** features_int = NULL;
	vector<MYINT**> features_intV(switches, NULL);
	float** features_float = NULL;

	// Initialize variables used for profiling.
	initializeProfiling();

	// Following variables are used for storing the results of the inference.
	vector<float*> vector_float_res;
	vector<int32_t*> vector_int_res;
	vector<int32_t**> vector_int_resV;
	vector<int32_t*> labelsInt;
	vector<float*> labelsFloat;
	list<thread> threads;

	MYINT*** features_intV_copy;

	string line1, line2;
	int counter = 0;

	if (version == Float) {
		profilingEnabled = true;
	}

	// Each iteration takes care of one datapoint.
	while (getline(featuresFile, line1) && getline(lablesFile, line2)) {
		// Read the feature vector and class ID.
		vector<string> features = getFeatures(line1);
		vector<string> labelString = getLabel(line2);
		int32_t* labelInt = new int32_t[numOutputs];
		float* labelFloat = new float[numOutputs];

		if (problem == Classification) {
			for (int i = 0; i < numOutputs; i++) {
				labelInt[i] = atoi(labelString[i].c_str());
			}
		} else if (problem == Regression) {
			for (int i = 0; i < numOutputs; i++) {
				labelFloat[i] = atof(labelString[i].c_str());
			}
		}

		// Allocate memory to store the feature vector as arrays.
		if (alloc == false) {
			features_size = (int)features.size();

			features_int = new MYINT* [features_size];
			for (int i = 0; i < features_size; i++) {
				features_int[i] = new MYINT[1];
			}

			for (int i = 0; i < switches; i++) {
				features_intV[i] = new MYINT* [features_size];
				for (int j = 0; j < features_size; j++) {
					features_intV[i][j] = new MYINT[1];
				}
			}

			features_float = new float* [features_size];
			for (int i = 0; i < features_size; i++) {
				features_float[i] = new float[1];
			}

			alloc = true;
		}

		// Populate the array using the feature vector.
		if (debugMode || version == Fixed) {
			populateFixedVector(features_int, features, scaleForX);
			for (int i = 0; i < switches; i++) {
				populateFixedVector(features_intV[i], features, scalesForX[i]);
			}
			populateFloatVector(features_float, features);
		} else {
			populateFloatVector(features_float, features);
		}

		// Invoke the predictor function.
		int* fixed_res = NULL;
		float* float_res = NULL;
		vector <int> resV(switches, -1);

		if (debugMode) {
			float_res = new float[numOutputs];
			seedotFloat(features_float, float_res);
			fixed_res = new int32_t[numOutputs];
			seedotFixed(features_int, fixed_res);
			//debug();
			vector_float_res.push_back(float_res);
			vector_int_res.push_back(fixed_res);
			if (problem == Classification) {
				labelsInt.push_back(labelInt);
			} else if (problem == Regression) {
				labelsFloat.push_back(labelFloat);
			}
			vector_int_resV.push_back(NULL);
		} else {
			// There are several codes generated which are built simultaneously.
			if (version == Fixed) {
				vector_float_res.push_back(new float[numOutputs]);
				vector_int_res.push_back(new int32_t[numOutputs]);
				// Populating labels for each generated code.
				if (problem == Classification) {
					labelsInt.push_back(labelInt);
				} else if (problem == Regression) {
					labelsFloat.push_back(labelFloat);
				}
				int** switchRes = new int* [switches];
				// Instantiating vectors for storing inference results for each generated code.
				for (int i = 0; i < switches; i++) {
					switchRes[i] = new int[numOutputs];
				}
				vector_int_resV.push_back(switchRes);
				// Instantiating vectors for storing features, integer and float.
				MYINT** features_int_copy = new MYINT* [features_size];
				for (int i = 0; i < features_size; i++) {
					features_int_copy[i] = new MYINT[1];
					features_int_copy[i][0] = features_int[i][0];
				}
				float** features_float_copy = new float* [features_size];
				for (int i = 0; i < features_size; i++) {
					features_float_copy[i] = new float[1];
					features_float_copy[i][0] = features_float[i][0];
				}
				features_intV_copy = new MYINT** [switches];
				for (int j = 0; j < switches; j++) {
					features_intV_copy[j] = new MYINT* [features_size];
					for (int i = 0; i < features_size; i++) {
						features_intV_copy[j][i] = new MYINT[1];
						features_intV_copy[j][i][0] = features_intV[j][i][0];
					}
				}
				// Launching one thread which processes one datapoint.
				if (threads.size() < 64) {
					threads.push_back(thread(launchThread, features_size, features_int_copy, features_intV_copy, features_float_copy, counter, vector_float_res.back(), vector_int_res.back(), vector_int_resV.back()));
				} else {
					threads.front().join();
					threads.pop_front();
					threads.push_back(thread(launchThread, features_size, features_int_copy, features_intV_copy, features_float_copy, counter, vector_float_res.back(), vector_int_res.back(), vector_int_resV.back()));
				}
			} else if (version == Float) {
				float_res = new float[numOutputs];
				seedotFloat(features_float, float_res);
				vector_float_res.push_back(float_res);
				vector_int_res.push_back(new int[numOutputs]);
				if (problem == Classification) {
					labelsInt.push_back(labelInt);
				} else if (problem == Regression) {
					labelsFloat.push_back(labelFloat);
				}
				vector_int_resV.push_back(NULL);
			}
		}

		if (!logProgramOutput) {
			output << "Inputs handled = " << counter + 1 << endl;
		}

		flushProfile();
		counter++;
	}

	for (list<thread>::iterator it = threads.begin(); it != threads.end(); it++) {
		it->join();
	}

	float disagreements = 0.0, reduced_disagreements = 0.0;

	// Correct, Disagreements are used for Classification problems' accuracy etc.
	// Errors, Ferrors are used for Regression problems' error etc.

	vector<int> correctV(switches, 0), totalV(switches, 0);
	vector<int> disagreementsV(switches, 0), reduced_disagreementsV(switches, 0);

	vector<float> errors(0, 0), ferrors(0, 0);
	vector<vector<float>> errorsV(switches, vector<float>(0, 0)), ferrorsV(switches, vector<float>(0, 0));

	ofstream trace("trace.txt");

	int correct = 0, total = 0;
	for (int i = 0; i < counter; i++) {
		int* fixed_res = vector_int_res[i];
		float* float_res = vector_float_res[i];
		int** resV = vector_int_resV[i];

		if (problem == Classification) {
			for (int j = 0; j < numOutputs; j++) {
				float res;
				if (version == Float) {
					res = float_res[j];
				} else {
					res = (float) fixed_res[j];
				}

				if (res != float_res[j]) {
					if (float_res[j] == labelsInt[i][j]) {
						reduced_disagreements++;
					}
					disagreements++;
				}

				if (res == labelsInt[i][j]) {
					correct++;
				} else {
					if (logProgramOutput) {
						output << "Main: Incorrect prediction for input " << total + 1 << " element " << j << ". Predicted " << res << " Expected " << labelsInt[i][j] << endl;
					}
				}
				total++;

				for (int k = 0; k < switches; k++) {
					if (version == Float) {
						throw "Multiple codes not expected in Floating point execution";
					}

					if (resV[k][j] != float_res[j]) {
						if (float_res[j] == labelsInt[i][j]) {
							reduced_disagreementsV[k]++;
						}
						disagreementsV[k]++;
					}

					if (resV[k][j] == labelsInt[i][j]) {
						correctV[k]++;
					} else {
						if (logProgramOutput) {
							output << "Sub "<< k <<": Incorrect prediction for input " << total + 1 << " element " << j << ". Predicted " << resV[k][j] << " Expected " << labelsInt[i][j] << endl;
						}
					}
					totalV[k]++;
				}
			}
		} else {
			for (int j = 0; j < numOutputs; j++) {
				float res;
				if (version == Float) {
					res = float_res[j];
				} else {
					res = ((float)fixed_res[j]) / ldexp(1.0, -scaleForY);
				}

				trace << res << " ";

				float error = 100.0 * fabs(res - labelsFloat[i][j]);
				float ferror = 100.0 * fabs(res - float_res[j]);
				errors.push_back(error);
				ferrors.push_back(ferror);
				total++;

				for (int k = 0; k < switches; k++) {
					if (version == Float) {
						throw "Multiple codes not expected in Floating point execution";
					}
					float normRes = ((float) resV[k][j]) / ldexp(1.0 , -scalesForY[k]);
					float error = 100.0 * fabs(normRes - labelsFloat[i][j]);
					float ferror = 100.0 * fabs(normRes - float_res[j]);
					errorsV[k].push_back(error);
					ferrorsV[k].push_back(ferror);
					totalV[k]++;
				}
			}
		}

		// Clearing memory.
		delete[] vector_int_res[i];
		delete[] vector_float_res[i];
		for (int k = 0; k < switches; k++) {
			delete[] vector_int_resV[i][k];
		}
		delete[] vector_int_resV[i];

		trace << endl;
	}

	trace.close();

	// Deallocate memory.
	for (int i = 0; i < features_size; i++) {
		delete features_int[i];
	}
	delete[] features_int;

	for (int i = 0; i < features_size; i++) {
		delete features_float[i];
	}
	delete[] features_float;

	for (int i = 0; i < switches; i++) {
		for (int j = 0; j < features_size; j++) {
			delete features_intV[i][j];
		}
		delete[] features_intV[i];
	}

	float accuracy = (float)correct / total * 100.0f;

	if ((argc == 6) && (argv[5] == "False"))
	{
		cout.precision(3);
		cout << fixed;
		cout << "\n\n#test points = " << total << endl;
		cout << "Correct predictions = " << correct << endl;
		cout << "Accuracy = " << accuracy << "\n\n";
	}

	output.precision(3);
	output << fixed;
	output << "\n\n#test points = " << total << endl;
	output << "Correct predictions = " << correct << endl;
	output << "Accuracy = " << accuracy << "\n\n";
	output.close();

	stats.precision(3);
	stats << fixed;
	stats << "default" << "\n";
	if (problem == Classification) {
		stats << accuracy << "\n";
		stats << ((float) disagreements) / numOutputs << "\n";
		stats << ((float) reduced_disagreements) / numOutputs << "\n";
	} else if (problem == Regression) {
		sort(errors.begin(), errors.end());
		sort(ferrors.begin(), ferrors.end());
		int index = 0.95 * errors.size() - 1;
		index = index > 0 ? index : 0;
		stats << errors[index] << "\n";
		stats << ferrors[index] << "\n";
		stats << "0.000\n";
	}

	if (version == Fixed) {
		for (int i = 0; i < switches; i++) {
			stats << i + 1 << "\n";
			if (problem == Classification) {
				stats << (float)correctV[i] / totalV[i] * 100.0f << "\n";
				stats << ((float) disagreementsV[i]) / numOutputs << "\n";
				stats << ((float) reduced_disagreementsV[i]) / numOutputs << "\n";
			} else if (problem == Regression) {
				sort(errorsV[i].begin(), errorsV[i].end());
				sort(ferrorsV[i].begin(), ferrorsV[i].end());
				int index = 0.95 * errorsV[i].size() - 1;
				index = index > 0 ? index : 0;
				stats << errorsV[i][index] << "\n";
				stats << ferrorsV[i][index] << "\n";
				stats << "0.000\n";
			}
		}
	}

	stats.close();

	if (version == Float) {
		dumpProfile();
	}

	if (datasetType == Training) {
		dumpRange(outputDir + "/profile.txt");
	}

	return 0;
}
