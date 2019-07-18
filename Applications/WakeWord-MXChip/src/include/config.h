/* 
 * Compilation Flgs for Configuration
 */

/*
 * We support 512 point FFT in F32 or Q31 data types.
 * Define one of the following to enable the corresponding
 * functions.
 * WARNING: Do not define here. Use compiler flags.
 */

// #define NFFT_256
// #define NFFT_512

/* The floating point type to use float32_t or q32_float.
 *
 * WARNING: Again, this file is not included in platformIO VSCode
 *  for some reason. Add the -DFPTYPE=float32_t flag as a temporary
 *  work around.
 */

// #define FP_F32
// #define FP_Q31