/**
   Dancing links implementation to solve exact cover problem
 */

#include <cstdlib>
#include <iostream>
#include <list>
#include <vector>
#include <string>

using namespace std;

#define LOOP_RIGHT(element, head) \
  for (Node *element = head -> right; element != head; element = element -> right)

#define LOOP_DOWN(element, head) \
  for (Node *element = head -> down; element != head; element = element -> down)


class DLX {

  // Number of elements in the universe and 
  // the number of subsets available
  int numElements;
  int numToBeCovered;
  int numSubsets;

  // Node structure
  struct Node {
    Node *left;
    Node *right;
    Node *up;
    Node *down;
    Node *col;
    Node *row;
    int colIndex;
    int subsets;
    string tag;
  };

  vector <Node*> colHeaders;
  list <Node*> solution;

  Node *header;

  /** Add node to the right of a list */
  void addRight(Node *head, Node * node) {
    node -> left = head -> left;
    node -> right = head;
    head -> left -> right = node;
    head -> left = node;
  }

  /** Add node to the bottom of a list */
  void addDown(Node *head, Node * node) {
    node -> up = head -> up;
    node -> down = head;
    head -> up -> down = node;
    head -> up = node;
  }

  /** Get element with minimum subsets */
  Node *getElementWithMinSubsets() {
    Node *min = header;
    LOOP_RIGHT(element, header) {
      if (element -> colIndex >= numToBeCovered) {
	break;
      }
      if (min -> subsets > element -> subsets) {
	min = element;
      }
    }
    return min;
  }

  /** Unlink a subset */
  void unlinkSubset(Node *node) {
    Node *subsetHead = node -> row;
    LOOP_RIGHT(element, subsetHead) {
      element -> up -> down = element -> down;
      element -> down -> up = element -> up;
      element -> col -> subsets --;
    }
    subsetHead -> up -> down = subsetHead -> down;
    subsetHead -> down -> up = subsetHead -> up;
  }

  /** Link subset */
  void linkSubset(Node * node) {
    Node *subsetHead = node -> row;
    LOOP_RIGHT(element, subsetHead) {
      element -> up -> down = element;
      element -> down -> up = element;
      element -> col -> subsets ++;
    }
    subsetHead -> up -> down = subsetHead;
    subsetHead -> down -> up = subsetHead;
  }

  /** Unlink column head */
  void unlinkColumn(Node *node) {
    Node *head = node -> col;
    head -> left -> right = head -> right;
    head -> right -> left = head -> left;
  }

  /** Link column head */
  void linkColumn(Node *node) {
    Node *head = node -> col;
    head -> left -> right = head;
    head -> right -> left = head;
  }



public:
  
  // Constructor
  DLX(int _numElements, int _numToBeCovered, int _numSubsets) {
    numElements = _numElements;
    numToBeCovered = _numToBeCovered;
    numSubsets = _numSubsets;

    // Setup header
    header = new Node;
    header -> left = header;
    header -> right = header;
    header -> down = header;
    header -> up = header;
    header -> subsets = numSubsets + 1;

    // Setup column headers
    colHeaders.resize(numElements, NULL);
    for (int i = 0; i < numElements; i ++) {
      Node *colHeader = new Node;
      addRight(header, colHeader);

      colHeader -> up = colHeader;
      colHeader -> down = colHeader;
      colHeader -> col = colHeader;
      colHeader -> colIndex = i;
      colHeader -> subsets = 0;
      
      colHeaders[i] = colHeader;
    }
  }


  /**
   * Insert a subset of elements into the DLX matrix
   * Assumes there exists atleast one element
   */
  void insertSubset(string tag, vector <int> elements) {
    Node *rowHead = new Node;
    addDown(header, rowHead);

    rowHead -> right = rowHead;
    rowHead -> left = rowHead;
    rowHead -> tag = tag;

    for (int i = 0; i < elements.size(); i ++) {
      Node *element = new Node;
      Node *colHead = colHeaders[elements[i]];

      addRight(rowHead, element);
      element -> row = rowHead;

      addDown(colHead, element);
      element -> col = colHead;

      colHead -> subsets ++;
    }
  }


  /**
   * Solve DLX. Returns true of solved, else false
   */
  bool solve(int level = 0) {

    // cerr << level << endl;
    // print();

    // if (level == 1) {
    //   return false
    // }
    
    // if no columns to cover, solved
    if (header -> right == header) return true;
    // if all columns to be covered are covered, solved
    if (header -> right -> colIndex >= numToBeCovered) return true;

    // get column with least ones
    Node *chosenElement = getElementWithMinSubsets();
    // cout << "Choosing element " << chosenElement -> colIndex << endl;

    // for each subset that covers that element
    LOOP_DOWN(chosenSubset, chosenElement) {

      // cout << "Choosing subset " << chosenSubset -> row -> tag << endl;

      // add the chosen subset to the solution
      Node *chosenSubsetHead = chosenSubset -> row;
      solution.push_back(chosenSubsetHead);

      list <Node *> unlinkedSubsets;
      
      // For each element in the chosen subset
      LOOP_RIGHT(element, chosenSubsetHead) {

	Node *elementHead = element -> col;
	// For all subsets that cover that element, unlink all those subsets
	LOOP_DOWN(subset, elementHead) {
	  unlinkedSubsets.push_back(subset);
	  unlinkSubset(subset);
	}

	// Unlink column
	unlinkColumn(elementHead);
      }

      if (solve(level + 1) == true) {
	return true;
      }

      // For each element in the chosen subset
      LOOP_RIGHT(element, chosenSubsetHead) {
	Node *elementHead = element -> col;
        // Link all unlinked subsets
	LOOP_DOWN(subset, elementHead) {
	  linkSubset(subset);
	}
	// Link column
	linkColumn(elementHead);	
      }

      for (list <Node *>::iterator it = unlinkedSubsets.begin(); it != unlinkedSubsets.end(); it ++) {
	linkSubset(*it);
      }

      // cout << "Backtrack subset " << chosenSubsetHead -> tag << endl << endl;
      // print();

      // Remove chosen subset from the solution
      solution.pop_back();
    }

    return false;    
  }

  /**
   * Print solution
   */
  void printSolution() {
    for (list <Node*>::iterator it = solution.begin(); it != solution.end(); it ++) {
      cout << (*it) -> tag << endl;
    }
  }
  

  /**
   * Print DLX stuff
   */
  void print() {
    int colIndex = 0;
    LOOP_DOWN(subset, header) {
      cout << subset -> tag << " "; 
    }
    cout << endl;
    LOOP_RIGHT(element, header) {
      cout << element -> colIndex << " ";
    }
    cout << endl;
    cout << endl;
  }
  
};
