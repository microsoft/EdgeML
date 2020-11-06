#include "dlx.h"

#include <string>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  DLX *solver;

  int numElements;
  int numToBeCovered;
  int numSubsets;
  
  cin >> numElements;
  cin >> numToBeCovered;
  cin >> numSubsets;

  solver = new DLX(numElements, numToBeCovered, numSubsets);

  for (int i = 0; i < numSubsets; i ++) {
    string tag;
    cin >> tag;

    int numElements;
    cin >> numElements;

    vector <int> elements;
    elements.resize(numElements);
    for (int j = 0; j < numElements; j ++) {
      cin >> elements[j];
    }

    solver -> insertSubset(tag, elements);
  }

  bool success = solver -> solve();
  if (success) {
    cerr << "Found solution " << endl;
    solver -> printSolution();
  } else {
    cerr << "Failed to find solution" << endl;
  }

  delete solver;
  
  return 0;
}
