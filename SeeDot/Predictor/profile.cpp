// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <limits>

#include "datatypes.h"
#include "profile.h"

using namespace std;

float m_all, M_all;
float m_exp, M_exp;

void initializeProfiling() {
	m_all = numeric_limits<float>::max();
	M_all = -numeric_limits<float>::max();

	m_exp = numeric_limits<float>::max();
	M_exp = -numeric_limits<float>::max();

	return;
}

void updateRange(float x) {
	if (x < m_all)
		m_all = x;
	if (x > M_all)
		M_all = x;
	return;
}

void updateRangeOfExp(float x) {
	if (x < m_exp)
		m_exp = x;
	if (x > M_exp)
		M_exp = x;
	return;
}

void dumpRange(string outputFile) {
	ofstream fout(outputFile);

	fout.precision(6);
	fout << fixed;
	fout << m_all << ", " << M_all << endl;
	fout << m_exp << ", " << M_exp << endl;

	return;
}

void diff(float* A, MYINT* B, MYINT scale, MYINT I, MYINT J) {

	float min = numeric_limits<float>::max(), max = 0, sum = 0;
	float min_relative = numeric_limits<float>::max(), max_relative = 0, sum_relative = 0;
	int count = 0;

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			float a = A[i * J + j];

			MYINT b = B[i * J + j];
			//float b_float = float(b) / scale;
			float b_float = float(ldexp(double(b), scale));

			float diff = abs(a - b_float);
			float diff_relative = diff / abs(a);

			if (diff < min)
				min = diff;
			if (diff > max) 
				max = diff;

			if (diff_relative < min_relative)
				min_relative = diff_relative;
			if (diff_relative > max_relative)
				max_relative = diff_relative;

			sum += diff;
			sum_relative += diff_relative;

			count++;
		}
	}

	float avg = sum / count;
	float avg_relative = sum_relative / count;

	cout << max << "\t" << avg << "\t" << min << "\t" << max_relative << "\t" << avg_relative << "\t" << min_relative << endl;

	return;
}

void diff(float* A, MYINT* B, MYINT scale, MYINT I, MYINT J, MYINT K) {

	float min = numeric_limits<float>::max(), max = 0, sum = 0;
	float min_relative = numeric_limits<float>::max(), max_relative = 0, sum_relative = 0;
	int count = 0;

	for (MYINT i = 0; i < I; i++) {
		for (MYINT j = 0; j < J; j++) {
			for (MYINT k = 0; k < K; k++) {
				float a = A[i * J * K + j * K + k];

				MYINT b = B[i * J * K + j * K + k];
				//float b_float = float(b) / scale;
				float b_float = float(ldexp(double(b), scale));

				float diff = abs(a - b_float);
				float diff_relative = diff / abs(a);

				if (diff < min)
					min = diff;
				if (diff > max)
					max = diff;

				if (diff_relative < min_relative)
					min_relative = diff_relative;
				if (diff_relative > max_relative)
					max_relative = diff_relative;

				sum += diff;
				sum_relative += diff_relative;

				count++;
			}
		}
	}

	float avg = sum / count;
	float avg_relative = sum_relative / count;

	cout << max << "\t" << avg << "\t" << min << "\t" << max_relative << "\t" << avg_relative << "\t" << min_relative << endl;

	return;
}
