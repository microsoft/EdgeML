// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __TIMER_H_
#define __TIMER_H_


#include <iomanip>
#include <ctime>
#include <chrono>

namespace EdgeML
{
  class Timer
  {
    static int level;
    std::clock_t before, after;
    std::chrono::time_point<std::chrono::system_clock> beforeSysT, afterSysT;
    std::string fn;
    int level_;

  public:
    Timer(std::string fn_name);
    ~Timer();

    float nextTime(const std::string& event_name);
  };
}
#endif
