#pragma once

#ifdef NDEBUG
  #include <nvToolsExt.h>

  #define startTrace(name) nvtxRangePush(name)
  #define endTrace(name) nvtxRangePop()

  #ifndef TRACE_LEVEL
    #define TRACE_LEVEL 4
  #endif

  #if TRACE_LEVEL >= 1
    #define startTrace1(name) startTrace(name)
  #else
    #define startTrace1(name)
  #endif

  #if TRACE_LEVEL >= 2
    #define startTrace2(name) startTrace(name)
  #else
    #define startTrace2(name)
  #endif

  #if TRACE_LEVEL >= 3
    #define startTrace3(name) startTrace(name)
  #else
    #define startTrace3(name)
  #endif

  #if TRACE_LEVEL >= 4
    #define startTrace4(name) startTrace(name)
  #else
    #define startTrace4(name)
  #endif

  #if TRACE_LEVEL >= 5
    #define startTrace5(name) startTrace(name)
  #else
    #define startTrace5(name)
  #endif

  #if TRACE_LEVEL >= 6
    #define startTrace6(name) startTrace(name)
  #else
    #define startTrace6(name)
  #endif

  #if TRACE_LEVEL >= 7
    #define startTrace7(name) startTrace(name)
  #else
    #define startTrace7(name)
  #endif

  #if TRACE_LEVEL >= 7
    #define startTrace7(name) startTrace(name)
  #else
    #define startTrace7(name)
  #endif

  #if TRACE_LEVEL >= 8
    #define startTrace8(name) startTrace(name)
  #else
    #define startTrace8(name)
  #endif

  #if TRACE_LEVEL >= 9
    #define startTrace9(name) startTrace(name)
  #else
    #define startTrace9(name)
  #endif


    #if TRACE_LEVEL >= 10
      #define startTrace10(name) startTrace(name)
    #else
      #define startTrace10(name)
    #endif

#else
  struct Trace { Trace(const char*) {} };

  #define startTrace(name)
  #define endTrace(name)
  #define startTrace1(name)
  #define startTrace2(name)
  #define startTrace3(name)
  #define startTrace4(name)
  #define startTrace5(name)
  #define startTrace6(name)
  #define startTrace7(name)
  #define startTrace8(name)
  #define startTrace9(name)
  #define startTrace10(name)
#endif
