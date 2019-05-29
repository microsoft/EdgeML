// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#pragma once

#define INT16
//#define INT32


#ifdef INT16
typedef int16_t MYINT;
#endif

#ifdef INT32
typedef int32_t MYINT;
#endif

typedef uint16_t MYUINT;
