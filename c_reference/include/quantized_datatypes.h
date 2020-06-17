#include <stdint.h>

// Macro for input type.
typedef int16_t MYINT;
// Macro for iterator type.
typedef uint16_t MYITE;
// Macro for intermediate buffer type.
typedef int32_t MYINM;
// Macro for scale variable type.
#ifdef SHIFT
  typedef uint8_t MYSCL;
#else
  typedef int16_t MYSCL;
#endif
// Macro for max value of input type.
#define MYINTMAX 32767
// Macro for min value of input type.
#define MYINTMIN -32768
