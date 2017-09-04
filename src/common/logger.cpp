// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <iostream>
#include "logger.h"
#include "blas_routines.h"

using namespace EdgeML;

int Logger::level = 0;

EdgeML::Logger::Logger(std::string fnName)
  :
  trace_print_func(NULL),
  info_print_func(NULL),
  warning_print_func(NULL),
  error_print_func(NULL),
  isTimerFileOpen(false),
  isDiagnosticFileOpen(false)
{
  fn = fnName;
  level_ = level;
  //level += 1;
#ifdef LIGHT_LOGGER
  std::string msg;
  for (int i = 0; i < level_; ++i) msg += "\t";
  msg += "Logging in " + fn;
  LOG_DIAGNOSTIC_MSG(msg);
#endif
}

EdgeML::Logger::Logger()
  :
  trace_print_func(NULL),
  info_print_func(NULL),
  warning_print_func(NULL),
  error_print_func(NULL),
  isTimerFileOpen(false),
  isDiagnosticFileOpen(false)
{}

void EdgeML::Logger::log_info(
  const std::string& msg,
  const std::string& fileName,
  const std::string& fnName,
  int lineNo)
{
  //std::string info_string
    //= fileName + "(" + fnName + ":" + std::to_string(lineNo) + "): " + msg;
  if (info_print_func)
    info_print_func(msg.c_str());
  else
    std::cout << msg << std::endl;
}

void EdgeML::Logger::log_trace(
  const std::string& msg,
  const std::string& fileName,
  const std::string& fnName,
  int lineNo)
{
  // If you dont wan't verbose information, switch this off.
  if (trace_print_func)
    trace_print_func(msg.c_str());
  else {
#ifdef VERBOSE 
    std::cout << msg << std::endl;
#endif
  }
}

void EdgeML::Logger::log_warning(
  const std::string& msg,
  const std::string& fileName,
  const std::string& fnName,
  int lineNo)
{
  std::string warning_string
    = "\n********* WARNING *********\n"
    + fileName + "(" + fnName + ":" + std::to_string(lineNo) + "): " + msg
    + "\n***************************\n";
  if (warning_print_func)
    warning_print_func(warning_string.c_str());
  else
    std::cerr << warning_string << std::endl;
}

void EdgeML::Logger::log_error(
  const std::string& msg,
  const std::string& fileName,
  const std::string& fnName,
  int lineNo)
{
  std::string error_string
    = "\n********** ERROR **********\n"
    + fileName + "(" + fnName + ":" + std::to_string(lineNo) + "): " + msg
    + "\n***************************\n";
  if (error_print_func)
    error_print_func(error_string.c_str());
  else
    std::cerr << error_string << std::endl;
}

void EdgeML::Logger::log_timer(
  const std::string& msg,
  const std::string& fileName,
  const std::string& fnName,
  int lineNo)
{
  if (isTimerFileOpen)
    timerLogStream << msg << std::endl;
}

bool EdgeML::Logger::openTimerLogFile(const std::string& dir)
{
  assert(!timerLogStream.is_open());
  assert(!isTimerFileOpen);

  timerLogStream.open(dir + "/timerLog", std::ofstream::out);
  if (timerLogStream.is_open())
    isTimerFileOpen = true;

  return  isTimerFileOpen;
}

bool EdgeML::Logger::openDiagnosticLogFile(const std::string& dir)
{
  assert(!diagnosticLogStream.is_open());
  assert(!isDiagnosticFileOpen);

  diagnosticLogStream.open(dir + "/diagnosticLog", std::ofstream::out);
  if (diagnosticLogStream.is_open())
    isDiagnosticFileOpen = true;

  return isDiagnosticFileOpen;
}

void EdgeML::Logger::log_diagnostic(
  const std::string& msg,
  const std::string& fileName,
  const std::string& fnName,
  int lineNo)
{
#ifdef LIGHT_LOGGER
  std::string str;
  for (int i = 0; i < level_; ++i) str += "\t";
  str += fileName + "(" + fnName + ":" + std::to_string(lineNo) + "): " + msg;
  if (isDiagnosticFileOpen)
    diagnosticLogStream << str << std::endl;
#endif
}

void EdgeML::Logger::log_diagnostic(
  const MatrixXuf& mat,
  const std::string& matName,
  const std::string& fnName,
  int lineNo)
{
#ifdef LOGGER
  std::string str;
  for (int i = 0; i < level_; ++i) str += "\t";
  str += "Min, Max, Norm of " + matName + "(@" + fnName + ": " + std::to_string(lineNo) + "): "
    + std::to_string(mat.minCoeff()) + ", " + std::to_string(mat.maxCoeff()) + ", "
    + std::to_string(mat.norm());
  if (isDiagnosticFileOpen)
    diagnosticLogStream << str << std::endl;
#endif
}

void EdgeML::Logger::log_diagnostic(
  const SparseMatrixuf& mat,
  const std::string& matName,
  const std::string& fnName,
  int lineNo)
{
#ifdef LOGGER
  std::string str;
  for (int i = 0; i < level_; ++i) str += "\t";
  str += "MaxAbs, Norm of " + matName + "(@" + fnName + ": " + std::to_string(lineNo) + "): "
    + std::to_string(maxAbsVal(mat)) + ", " + std::to_string(mat.norm());
  if (isDiagnosticFileOpen)
    diagnosticLogStream << str << std::endl;
#endif
}

void EdgeML::Logger::log_diagnostic(
  const int& num,
  const std::string& numName,
  const std::string& fnName,
  int lineNo)
{
#ifdef LOGGER
  std::string str;
  for (int i = 0; i < level_; ++i) str += "\t";
  str += "Value of " + numName + "(@" + fnName + ": " + std::to_string(lineNo) + "): "
    + std::to_string(num);
  if (isDiagnosticFileOpen)
    diagnosticLogStream << str << std::endl;
#endif
}

EdgeML::Logger::~Logger()
{
#ifdef LIGHT_LOGGER
  std::string msg;
  for (int i = 0; i < level_; ++i) msg += "\t";
  msg += "Done logging in " + fn;
  LOG_DIAGNOSTIC_MSG(msg);
#endif
  //level -= 1;
  if (isTimerFileOpen)
    timerLogStream.close();
  if (isDiagnosticFileOpen)
    diagnosticLogStream.close();
}

static EdgeML::Logger GlobalLogger;

void EdgeML::global_log_info(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_info(msg, fileName, fnName, lineNo);
}
void EdgeML::global_log_trace(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_trace(msg, fileName, fnName, lineNo);
}
void EdgeML::global_log_warning(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_warning(msg, fileName, fnName, lineNo);
}
void EdgeML::global_log_error(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_error(msg, fileName, fnName, lineNo);
}


void EdgeML::global_log_timer(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_timer(msg, fileName, fnName, lineNo);
}
bool EdgeML::global_openTimerLogFile(const std::string& dir)
{
  return GlobalLogger.openTimerLogFile(dir);
}

void EdgeML::global_log_diagnostic(const std::string& msg, const std::string& fileName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_diagnostic(msg, fileName, fnName, lineNo);
}
void EdgeML::global_log_diagnostic(const MatrixXuf& mat, const std::string& matName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_diagnostic(mat, matName, fnName, lineNo);
}
void EdgeML::global_log_diagnostic(const SparseMatrixuf& mat, const std::string& matName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_diagnostic(mat, matName, fnName, lineNo);
}
void EdgeML::global_log_diagnostic(const int& num, const std::string& numName, const std::string& fnName, int lineNo)
{
  GlobalLogger.log_diagnostic(num, numName, fnName, lineNo);
}
bool EdgeML::global_openDiagnosticLogFile(const std::string& dir)
{
  return GlobalLogger.openDiagnosticLogFile(dir);
}

void EdgeML::global_set_trace_func(ChannelFunc func_)
{
  GlobalLogger.set_trace_func(func_);
}
void EdgeML::global_set_info_func(ChannelFunc func_)
{
  GlobalLogger.set_info_func(func_);
}
void EdgeML::global_set_warning_func(ChannelFunc func_)
{
  GlobalLogger.set_warning_func(func_);
}
void EdgeML::global_set_error_func(ChannelFunc func_)
{
  GlobalLogger.set_error_func(func_);
}
