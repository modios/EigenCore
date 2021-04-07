// EigenNative.h : Include file for standard system include files,
// or project specific include files.

#pragma once

#include <limits>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <iostream>
#define UNUSED(x) (void)(x)
#define DEBUG_ONLY(x) (void)(x)
#define MIN(a,b) (((a)<(b))?(a):(b))

#if defined _WIN32 || defined _WIN64
#include <intrin.h>
#include <sal.h>
#define EXPORT_API(ret) extern "C" __declspec(dllexport) ret
#else
#include "UnixSal.h"
#define EXPORT_API(ret) extern "C" __attribute__((visibility("default"))) ret
#define __forceinline __attribute__((always_inline)) inline
#endif


