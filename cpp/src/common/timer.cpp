// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "timer.h"
#include "logger.h"
#include <ctime>
#include <iostream>
#include <string>

#ifdef CONCISE
float thresh = 0.1f;
#else
float thresh = -0.0001f;
#endif

using namespace EdgeML;

int Timer::level = 0;  // STATIC INITIALIZATION

EdgeML::Timer::Timer(std::string fn_name)
{
  fn = fn_name;
  level_ = level;
  level += 1;

#ifdef TIMER
  std::string indentLevel;
  for (int i = 0; i < level_; ++i) indentLevel += "\t";
  LOG_TIMER(indentLevel + "Starting timer in " + fn);
#endif

  before = after = std::clock();
  //clock_gettime (CLOCK_MONOTONIC, &before_wall);
  afterSysT = beforeSysT = std::chrono::system_clock::now();
}

EdgeML::Timer::~Timer()
{
#ifdef TIMER
  nextTime("returning");
#endif
  level -= 1;
}

float EdgeML::Timer::nextTime(const std::string& event_name)
{
  after = std::clock();
  afterSysT = std::chrono::system_clock::now();

  auto delta = ((float)after - (float)before) / (float)CLOCKS_PER_SEC;
  std::chrono::duration<double> deltaSysT = afterSysT - beforeSysT;

  before = after;
  beforeSysT = afterSysT;

#ifdef TIMER
  if (deltaSysT.count() > thresh) {
    std::string indentLevel;
    for (int i = 0; i < level_; ++i) indentLevel += "\t";
    std::stringstream outputString;
    outputString << indentLevel << "Time for " << std::setw(47) << std::left << event_name
      << delta << "s (proc) "
      << deltaSysT.count() << "s (wall)";
    LOG_TIMER(outputString.str());
  }
#endif
  return delta;
}
