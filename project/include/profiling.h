#pragma once

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
#define endTrace1(name)
#define endTrace2(name)
#define endTrace3(name)
#define endTrace4(name)
#define endTrace5(name)
#define endTrace6(name)
#define endTrace7(name)
#define endTrace8(name)
#define endTrace9(name)
#define endTrace10(name)

#ifdef TRACE_LEVEL
  #include <nvToolsExt.h>

  #undef startTrace
  #undef endTrace
  #define startTrace(name) nvtxRangePush(name)
  #define endTrace(name) nvtxRangePop()

  #if TRACE_LEVEL >= 1
    #undef startTrace1
    #undef endTrace1
    #define startTrace1(name) startTrace(name)
    #define endTrace1() endTrace()
  #endif

  #if TRACE_LEVEL >= 2
    #undef startTrace2
    #undef endTrace2
    #define startTrace2(name) startTrace(name)
    #define endTrace2() endTrace()
  #endif

  #if TRACE_LEVEL >= 3
    #undef startTrace3
    #undef endTrace3
    #define startTrace3(name) startTrace(name)
    #define endTrace3() endTrace()
  #endif

  #if TRACE_LEVEL >= 4
    #undef startTrace4
    #undef endTrace4
    #define startTrace4(name) startTrace(name)
    #define endTrace4() endTrace()
  #endif

  #if TRACE_LEVEL >= 5
    #undef startTrace5
    #undef endTrace5
    #define startTrace5(name) startTrace(name)
    #define endTrace5() endTrace()
  #endif

  #if TRACE_LEVEL >= 6
    #undef startTrace6
    #undef endTrace6
    #define startTrace6(name) startTrace(name)
    #define endTrace6() endTrace()
  #endif

  #if TRACE_LEVEL >= 7
    #undef startTrace7
    #undef endTrace7
    #define startTrace7(name) startTrace(name)
    #define endTrace7() endTrace()
  #endif

  #if TRACE_LEVEL >= 8
    #undef startTrace8
    #undef endTrace8
    #define startTrace8(name) startTrace(name)
    #define endTrace8() endTrace()
  #endif

  #if TRACE_LEVEL >= 9
    #undef startTrace9
    #undef endTrace9
    #define startTrace9(name) startTrace(name)
    #define endTrace9() endTrace()
  #endif

  #if TRACE_LEVEL >= 10
    #undef startTrace10
    #undef endTrace10
    #define startTrace10(name) startTrace(name)
    #define endTrace10() endTrace()
  #endif

#endif
