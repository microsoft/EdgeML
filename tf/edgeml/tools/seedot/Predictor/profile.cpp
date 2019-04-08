// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include <fstream>
#include <limits>

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
