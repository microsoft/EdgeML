// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cassert>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <sys/stat.h>
#include <fcntl.h>

#ifdef LINUX 
#include <unistd.h>
#include <sys/mman.h>
#endif

#ifdef WINDOWS
#include <Windows.h>
#include <FileAPI.h>
#include <Winbase.h>
#endif

#include "mmaped.h"

#define INF 1000000000

using namespace EdgeML::FileIO;

Data::Data(
  std::string filename,
  MatrixXuf& data,
  MatrixXuf& label,
  dataCount_t max_entries,
  featureCount_t _COL_LABEL,
  featureCount_t _COL_FEATURE,
  featureCount_t _NUM_COLS,
  featureCount_t _NUM_FEATURES,
  featureCount_t _NUM_LABELS,
  EdgeML::DataFormat& formatType)
{
  COL_LABEL = _COL_LABEL;
  COL_FEATURE = _COL_FEATURE;
  NUM_COLS = _NUM_COLS;
  NUM_FEATURES = _NUM_FEATURES;
  NUM_LABELS = _NUM_LABELS;

  if (formatType != EdgeML::libsvmFormat) {
    assert(NUM_FEATURES <= NUM_COLS);
    assert(COL_LABEL <= COL_FEATURE);
    assert(COL_FEATURE + NUM_FEATURES == NUM_COLS);
  }

#ifdef LINUX 
  const char* file_char_str = filename.c_str();
  int fd = open(file_char_str, O_RDONLY);
  if (!(fd > 0)) {
    LOG_ERROR("Data file " + filename + " not found. Program will stop now.");
    assert(false);
  }
  struct stat sb;
  assert(fstat(fd, &sb) == 0);
  off_t fileSize = sb.st_size;
  assert(sizeof(off_t) == 8);
  void *buf = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(buf);
  assert(sizeof(dataCount_t) == sizeof(off_t));

  if (formatType == EdgeML::libsvmFormat)
    int nRead = libsvmFillEntries((char*)buf, data, label, max_entries,
      NUM_COLS, fileSize, formatType);
  else
    int nRead = fillEntries((char*)buf, data, label, max_entries,
      NUM_COLS, fileSize, formatType);
  assert(munmap(buf, fileSize) == 0);
  close(fd);
#endif
#ifdef WINDOWS
  HANDLE hFile =
    CreateFileA(filename.c_str(),
      GENERIC_READ, FILE_SHARE_READ, NULL, // default security
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  assert(hFile != INVALID_HANDLE_VALUE);

  LARGE_INTEGER fileSizeLI;
  GetFileSizeEx(hFile, &fileSizeLI);
  //assert(fileSizeLI.HighPart == 0); // Can map up to 2GB files??
  uint64_t fileSize = fileSizeLI.LowPart + (((uint64_t)fileSizeLI.HighPart) << 32);

  HANDLE hMapFile =
    CreateFileMapping(hFile,
      NULL, PAGE_READONLY,// default security & READ/WRITE mode
      0, 0,// maximum object sizes (high-order, lower-order DWORD)
      NULL);// name of mapping object
  assert(hMapFile != NULL);

  // 3rd and 4th params are higher and lower order bits of start offset
  // 5th param = 0 for mapping entire file.
  char* pBuf =
    (char*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
  assert(pBuf != NULL);


  if (formatType == EdgeML::libsvmFormat)
    int nRead = libsvmFillEntries((char*)pBuf, data, label, max_entries,
      NUM_COLS, fileSize, formatType);
  else
    int nRead = fillEntries((char*)pBuf, data, label, max_entries,
      NUM_COLS, fileSize, formatType);

  CloseHandle(hMapFile);
  CloseHandle(hFile);
#endif
}

Data::Data(
  std::string filename,
  SparseMatrixuf& data,
  SparseMatrixuf& label,
  dataCount_t max_entries,
  featureCount_t _COL_LABEL,
  featureCount_t _COL_FEATURE,
  featureCount_t _NUM_COLS,
  featureCount_t _NUM_FEATURES,
  featureCount_t _NUM_LABELS,
  EdgeML::DataFormat& formatType)
{
  COL_LABEL = _COL_LABEL;
  COL_FEATURE = _COL_FEATURE;
  NUM_COLS = _NUM_COLS;
  NUM_FEATURES = _NUM_FEATURES;
  NUM_LABELS = _NUM_LABELS;
  if (filename.empty()) { data = SparseMatrixuf(0, 0); label = SparseMatrixuf(0, 0); return;  }

#ifdef LINUX 
  if (formatType != EdgeML::libsvmFormat) {
    assert(NUM_FEATURES <= NUM_COLS);
    assert(COL_LABEL <= COL_FEATURE);
    assert(COL_FEATURE + NUM_FEATURES == NUM_COLS);
  }

  const char* file_char_str = filename.c_str();
  int fd = open(file_char_str, O_RDONLY);
  if (!(fd > 0)) {
    LOG_ERROR("Data file " + filename + " not found. Program will stop now.");
    assert(false);
  }
  struct stat sb;
  assert(fstat(fd, &sb) == 0);
  off_t fileSize = sb.st_size;
  assert(sizeof(off_t) == 8);
  void *buf = mmap(NULL, fileSize, PROT_READ, MAP_PRIVATE, fd, 0);
  assert(buf);
  assert(sizeof(dataCount_t) == sizeof(off_t));

  assert(formatType == EdgeML::libsvmFormat);
  int nRead = libsvmFillEntries((char*)buf, data, label,
    max_entries, NUM_COLS, fileSize,
    formatType);
  assert(munmap(buf, fileSize) == 0);
  close(fd);
#endif

#ifdef _MSC_VER
  HANDLE hFile =
    CreateFileA(filename.c_str(),
      GENERIC_READ, FILE_SHARE_READ, NULL, // default security
      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
  assert(hFile != INVALID_HANDLE_VALUE);

  LARGE_INTEGER fileSizeLI;
  GetFileSizeEx(hFile, &fileSizeLI);
  //assert(fileSizeLI.HighPart == 0); // Can map up to 2GB files??
  uint64_t fileSize = fileSizeLI.LowPart + (((uint64_t)fileSizeLI.HighPart) << 32);

  HANDLE hMapFile =
    CreateFileMapping(hFile,
      NULL, PAGE_READONLY,// default security & READ/WRITE mode
      0, 0,// maximum object sizes (high-order, lower-order DWORD)
      NULL);// name of mapping object
  assert(hMapFile != NULL);

  // 3rd and 4th params are higher and lower order bits of start offset
  // 5th param = 0 for mapping entire file.
  char* pBuf =
    (char*)MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
  assert(pBuf != NULL);

  assert(formatType == EdgeML::libsvmFormat);
  int nRead = libsvmFillEntries((char*)pBuf, data, label,
    max_entries, NUM_COLS, fileSize,
    formatType);

  CloseHandle(hMapFile);
  CloseHandle(hFile);
#endif  
}



//Input: @max_entries: max lines of data you want to read
//Input: @num_cols: Expect each line to have precisely these number of columns
//Input: @buf: mmaped buffer, read only
//Input: @fileSize: number of bytes that have been mmap'd
// Return the number of entries read.
size_t Data::fillEntries(
  char*buf,
  MatrixXuf& data,
  MatrixXuf& label,
  dataCount_t max_entries,
  featureCount_t num_cols,
  uint64_t fileSize,
  EdgeML::DataFormat& formatType)
{
  data = MatrixXuf::Zero(NUM_FEATURES, max_entries);
  label = MatrixXuf::Zero(NUM_LABELS, max_entries);

  FP_TYPE value = 0;
  LABEL_TYPE lab = 0;
  int dec = -INF; // Need to keep track of number of decimal points. Seamlessly works with int also. 
  off_t nRead = 0; // How many lines have we read?
  featureCount_t col = 0; // Which column are we trying to read?
  bool is_positive = true;
  bool exp_flag = false;
  bool exp_is_positive = true;
  bool nan_flag = false;
  int exp_val = 0;

  for (off_t i = 0; i < fileSize; ++i) { // Iterate over chars in a file
    if (nRead == max_entries) break;
    assert(col <= num_cols);
    if (col == COL_LABEL) {
      if (formatType == EdgeML::tsvFormat) {
        switch (buf[i]) {
        case '\t':
#ifdef ZERO_BASED_IO
          lab++;
#endif	  
          if (!(lab > 0 && lab <= NUM_LABELS)) {
            LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
              + "Label value = " + std::to_string(lab) + " incompatible with data-format restrictions specied in README."
              + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
            assert(false);
          }
          label(lab - 1, nRead) = 1.0; lab = 0; col++;
          break;
        case '0': case '1': case '2': case '3':case '4':
        case '5': case '6': case '7': case '8': case '9':
          lab *= 10; lab += buf[i] - '0';
          break;
        default:
          LOG_ERROR("Bad format in line: " + std::to_string(nRead) + "; character read: '" + std::string(1, buf[i]) + "'");
        }
      }
    }
    else if (col < COL_FEATURE) {
      switch (buf[i]) {
      case '\t':
        col++;
      default:
        lab = 0;
        break;
      }
    }
    else if (col >= COL_FEATURE) {
      switch (buf[i]) {
      case '\r':
        //	assert(col == num_cols-1);
        break;
      case '\n':
        if (exp_flag)
          value = value * (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
        if (!is_positive)
          value *= -1;
        if (dec > 0)
          value *= (FP_TYPE)pow(0.1, dec);

        data(col - COL_FEATURE, nRead) = value;

        if (col != num_cols - 1)
          for (auto c = col + 1; c < num_cols; ++c)
            data(c - COL_FEATURE, nRead) = 0;

        exp_flag = false; is_positive = true; dec = -INF;
        value = 0.0f; col = 0;	nRead++;
        break;

      case '\t':
        if (exp_flag)
          value *= (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
        if (!is_positive)
          value *= -1;
        if (dec > 0)
          value *= (FP_TYPE)pow(0.1, dec);

        data(col - COL_FEATURE, nRead) = value;

        exp_flag = false; is_positive = true; dec = -INF;

        dec = -INF;
        value = 0.0f;
        col++;
        break;

      case '-':
        if (exp_flag)
          exp_is_positive = false;
        else
          is_positive = false;
        break;

      case 'e':
        exp_flag = true; exp_val = 0; exp_is_positive = true;
        break;

      case 'N':
      case 'a':
        if (nan_flag == 0)
          LOG_WARNING("NaN possibly exists in the file. This code is not designed to handle NaN elaborately, and works only in specific cases.");
        nan_flag = 1;
        break;

      case 'I':
      case 'n':
      case 'f':
        if (nan_flag == 0)
          LOG_WARNING("Inf possibly exists in the file. This code is not designed to handle Inf elaborately, and works only in specific cases.");
        nan_flag = 1;
        break;

      case '0': case '1': case '2': case '3':case '4':
      case '5': case '6': case '7': case '8': case '9':
        if (exp_flag) {
          exp_val *= 10; exp_val += buf[i] - '0';
          break;
        }
        value *= 10.0f; value += buf[i] - '0';
        dec++;
        break;

      case '.':
        dec = 0;
        break;

      default:
        exp_flag = false; is_positive = true; exp_is_positive = true;
        value = 0.0f; exp_val = 0;  dec = -INF;
        col = lab = 0;
        LOG_ERROR("Bad format in line: " + std::to_string(nRead) + "; character read: '" + std::string(1, buf[i]) + "'");

        while (buf[i] != '\n')
          i++;
        i++;
      }
    }
  }
  if ((nRead != max_entries) && (col == num_cols - 1)) { // Didnt reach "\n" on last line
    if (exp_flag)
      value = value * (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
    if (!is_positive)
      value *= -1.0f;
    if (dec > 0)
      value *= (FP_TYPE)pow(0.1, dec);

    data(NUM_FEATURES - 1, nRead) = value;

    nRead++;
  }
  assert(nRead <= max_entries && "more entries in file than specified");

  data.conservativeResize(NUM_FEATURES, nRead);
  label.conservativeResize(NUM_LABELS, nRead);

  LOG_INFO("#Lines of data read: " + std::to_string(nRead));
  return nRead;
}

//Input: @max_entries: max lines of data you want to read
//Input: @num_cols: Expect each line to have precisely these number of columns
//Input: @buf: mmaped buffer, read only
//Input: @fileSize: number of bytes that have been mmap'd
// Return the number of entries read.
size_t Data::libsvmFillEntries(char*buf,
  MatrixXuf& data,
  MatrixXuf& label,
  dataCount_t max_entries,
  featureCount_t num_cols,
  uint64_t fileSize,
  EdgeML::DataFormat& format_type)
{
  data = MatrixXuf::Zero(NUM_FEATURES, max_entries);
  label = MatrixXuf::Zero(NUM_LABELS, max_entries);

  FP_TYPE value = 0;
  int dec = -INF; // Need to keep track of number of decimal points. Seamlessly works with int also. 
  off_t nRead = 0; // How many lines have we read?
  featureCount_t col = 0; // Which column are we trying to read?
  bool is_positive = true;
  bool index_flag = false;
  int index_value = 0;
  bool exp_flag = false;
  bool exp_is_positive = true;
  bool nan_flag = false;
  int exp_val = 0;

  for (off_t i = 0; i < fileSize; ++i) { // Iterate over chars in a file
    if (nRead == max_entries) break;
    assert(col <= num_cols);
    switch (buf[i]) {
    case '\r':
      //	assert(col == num_cols-1);
      break;
    case '\n':
      if (exp_flag)
        value = value * (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
      if (!is_positive)
        value *= -1;
      if (dec > 0)
        value *= (FP_TYPE)pow(0.1, dec);

      assert(index_flag == true);

#ifdef ZERO_BASED_IO
      index_value++;
#endif	    
      if (!(index_value > 0 && index_value <= NUM_FEATURES)) {
        LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
          + "Index value = " + std::to_string(index_value) + " incompatible with data-format restrictions specied in README."
          + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
        assert(false);
      }
      data(index_value - 1, nRead) = value;

      exp_flag = false; is_positive = true; dec = -INF;
      index_flag = false; index_value = 0;
      value = 0; col = 0;	nRead++;
      break;

    case ':':
      assert(index_flag == false);
      index_value = (int)std::round(value);
      index_flag = true;
      value = 0.0f;
      break;

    case '\t': case ' ': case ',':
      if (exp_flag)
        value *= (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
      if (!is_positive)
        value *= -1;
      if (dec > 0)
        value *= (FP_TYPE)pow(0.1, dec);

      if (index_flag == true) {
#ifdef ZERO_BASED_IO
        index_value++;
#endif	    
        if (!(index_value > 0 && index_value <= NUM_FEATURES)) {
          LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
            + "Index value = " + std::to_string(index_value) + " incompatible with data-format restrictions specied in README."
            + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
          assert(false);
        }
        data(index_value - 1, nRead) = value;
        index_flag = false; index_value = 0;
      }
      else {
        labelCount_t labelValue = (labelCount_t)std::round(value);

#ifdef ZERO_BASED_IO
        labelValue++;
#endif	  
        if (!(labelValue > 0 && labelValue <= NUM_LABELS)) {
          LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
            + "Label value = " + std::to_string(labelValue) + " incompatible with data-format restrictions specied in README."
            + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
          assert(false);
        }
        label(labelValue - 1, nRead) = 1.0;

        assert((dec < 0) && (exp_flag == false) && (is_positive == true));
      }
      exp_flag = false; is_positive = true; dec = -INF;

      dec = -INF;
      value = 0.0f;
      col++;
      break;

    case '-':
      if (exp_flag)
        exp_is_positive = false;
      else
        is_positive = false;
      break;

    case 'e':
      exp_flag = true; exp_val = 0; exp_is_positive = true;
      break;

    case 'N':
    case 'a':
      if (nan_flag == 0)
        LOG_WARNING("NaN possibly exists in the file. This code is not designed to handle NaN elaborately, and works only in specific cases.");
      nan_flag = 1;
      break;

    case 'I':
    case 'n':
    case 'f':
      if (nan_flag == 0)
        LOG_WARNING("Inf possibly exists in the file. This code is not designed to handle Inf elaborately, and works only in specific cases.");
      nan_flag = 1;
      break;

    case '0': case '1': case '2': case '3':case '4':
    case '5': case '6': case '7': case '8': case '9':
      if (exp_flag) {
        exp_val *= 10; exp_val += buf[i] - '0';
        break;
      }
      value *= 10; value += buf[i] - '0';
      dec++;
      break;

    case '.':
      dec = 0;
      break;

    default:
      exp_flag = false; is_positive = true; exp_is_positive = true;
      value = 0.0f; exp_val = 0;  dec = -INF;
      col = 0;
      LOG_ERROR("Bad format in line: " + std::to_string(nRead) + "; character read: '" + std::string(1, buf[i]) + "'");

      while (buf[i] != '\n')
        i++;
      i++;
    }
  }

  if ((nRead != max_entries) && (col == num_cols - 1)) { // Didnt reach "\n" on last line
    if (exp_flag)
      value = value * (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
    if (!is_positive)
      value *= -1;
    if (dec > 0)
      value *= (FP_TYPE)pow(0.1, dec);

#ifdef ZERO_BASED_IO
    index_value++;
#endif
    if (!(index_value > 0 && index_value <= NUM_FEATURES)) {
      LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
        + "Index value = " + std::to_string(index_value) + " incompatible with data-format restrictions specied in README."
        + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
      assert(false);
    }
    data(index_value - 1, nRead) = value;

    nRead++;
  }
  assert(nRead <= max_entries && "more entries in file than specified");

  data.conservativeResize(NUM_FEATURES, nRead);
  label.conservativeResize(NUM_LABELS, nRead);

  LOG_INFO("#Lines of data read: " + std::to_string(nRead) + "\n");
  return nRead;
}

//Input: @max_entries: max lines of data you want to read
//Input: @num_cols: Expect each line to have precisely these number of columns
//Input: @buf: mmaped buffer, read only
//Input: @fileSize: number of bytes that have been mmap'd
// Return the number of entries read.
size_t Data::libsvmFillEntries(char*buf,
  SparseMatrixuf& data,
  SparseMatrixuf& label,
  dataCount_t max_entries,
  featureCount_t num_cols,
  uint64_t fileSize,
  EdgeML::DataFormat& format_type)
{
  std::vector <Trip> data_triplet;
  std::vector <Trip> label_triplet;

  FP_TYPE value = 0;
  int dec = -INF; // Need to keep track of number of decimal points. Seamlessly works with int also. 
  off_t nRead = 0; // How many lines have we read?
  featureCount_t col = 0; // Which column are we trying to read?
  bool is_positive = true;
  bool index_flag = false;
  int index_value = 0;
  bool exp_flag = false;
  bool exp_is_positive = true;
  bool nan_flag = false;
  int exp_val = 0;

  for (off_t i = 0; i < fileSize; ++i) { // Iterate over chars in a file
    if (nRead == max_entries) break;
    assert(col <= num_cols);
    switch (buf[i]) {
    case '\r':
      //	assert(col == num_cols-1);
      break;
    case '\n':
      if (index_flag == true) {

        if (exp_flag)
          value = value * (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
        if (!is_positive)
          value *= -1;
        if (dec > 0)
          value *= (FP_TYPE)pow(0.1, dec);

#ifdef ZERO_BASED_IO
        index_value++;
#endif	    	  
        if (!(index_value > 0 && index_value <= NUM_FEATURES)) {
          LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
            + "Index value = " + std::to_string(index_value) + " incompatible with data-format restrictions specied in README."
            + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
          assert(false);
        }
        data_triplet.push_back(Trip(index_value - 1, nRead, value));
      }

      exp_flag = false; is_positive = true; dec = -INF;
      index_flag = false; index_value = 0;
      value = 0; col = 0;	nRead++;
      break;

    case ':':
      assert(index_flag == false);
      index_value = (labelCount_t)std::round(value);
      index_flag = true;
      value = 0;
      break;

    case '\t': case ' ': case ',':
      if (exp_flag)
        value *= (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
      if (!is_positive)
        value *= -1;
      if (dec > 0)
        value *= (FP_TYPE)pow(0.1, dec);

      if (index_flag == true) {
#ifdef ZERO_BASED_IO
        index_value++;
#endif	    
        if (!(index_value > 0 && index_value <= NUM_FEATURES)) {
          LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
            + "Index value = " + std::to_string(index_value) + " incompatible with data-format restrictions specied in README."
            + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
          assert(false);
        }
        data_triplet.push_back(Trip(index_value - 1, nRead, value));
        index_flag = false; index_value = 0;
      }
      else {
        labelCount_t labelValue = (labelCount_t)std::round(value);

#ifdef ZERO_BASED_IO
        labelValue++;
#endif
        if (!(labelValue > 0 && labelValue <= NUM_LABELS)) {
          LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
            + "Label value = " + std::to_string(labelValue) + " incompatible with data-format restrictions specied in README."
            + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
          assert(false);
        }
        label_triplet.push_back(Trip(labelValue - 1, nRead, 1.0f));

        assert((dec < 0) && (exp_flag == false) && (is_positive == true));
      }
      exp_flag = false; is_positive = true; dec = -INF;

      dec = -INF;
      value = 0;
      col++;
      break;

    case '-':
      if (exp_flag)
        exp_is_positive = false;
      else
        is_positive = false;
      break;

    case 'e':
      exp_flag = true; exp_val = 0; exp_is_positive = true;
      break;

    case 'N':
    case 'a':
      if (nan_flag == 0)
        LOG_WARNING("NaN possibly exists in the file. This code is not designed to handle Inf elaborately, and works only in specific cases.");
      nan_flag = 1;
      break;

    case 'I':
    case 'n':
    case 'f':
      if (nan_flag == 0)
        LOG_WARNING("Inf possibly exists in the file. This code is not designed to handle Inf elaborately, and works only in specific cases.");
      nan_flag = 1;
      break;

    case '0': case '1': case '2': case '3':case '4':
    case '5': case '6': case '7': case '8': case '9':
      if (exp_flag) {
        exp_val *= 10; exp_val += buf[i] - '0';
        break;
      }
      value *= 10; value += buf[i] - '0';
      dec++;
      break;

    case '.':
      dec = 0;
      break;

    default:
      exp_flag = false; is_positive = true; exp_is_positive = true;
      value = 0.0f; exp_val = 0;  dec = -INF;
      col = 0;
      LOG_ERROR("Bad format in line: " + std::to_string(nRead) + "; character read: '" + std::string(1, buf[i]) + "'");

      while (buf[i] != '\n')
        i++;
      i++;
    }
  }

  if ((nRead != max_entries) && (col == num_cols - 1)) { // Didnt reach "\n" on last line
    if (exp_flag)
      value = value * (FP_TYPE)pow(10, exp_val*(exp_is_positive ? 1 : -1));
    if (!is_positive)
      value *= -1;
    if (dec > 0)
      value *= (FP_TYPE)pow(0.1, dec);

#ifdef ZERO_BASED_IO
    index_value++;
#endif
    if (!(index_value > 0 && index_value <= NUM_FEATURES)) {
      LOG_ERROR("Error in line " + std::to_string(nRead) + " of input file.\n"
        + "Index value = " + std::to_string(index_value) + " incompatible with data-format restrictions specied in README."
        + "\nCheck also if ZERO_BASED_IO flag is set or not (in config.mk or elsewhere).");
      assert(false);
    }
    data_triplet.push_back(Trip(index_value - 1, nRead, value));
    nRead++;
  }
  assert(nRead <= max_entries && "more entries in file than specified");

  data = SparseMatrixuf(NUM_FEATURES, nRead);
  label = SparseMatrixuf(NUM_LABELS, nRead);
  LOG_INFO("Number of non-zero entries in data-matrix = " + std::to_string(data_triplet.size()));
  LOG_INFO("Number of non-zero entries in label-matrix = " + std::to_string(label_triplet.size()));

  data.setFromTriplets(data_triplet.begin(), data_triplet.end());
  label.setFromTriplets(label_triplet.begin(), label_triplet.end());

  LOG_INFO("#Lines of data read: " + std::to_string(nRead) + "\n");
  return nRead;
}
