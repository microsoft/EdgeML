// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "quantized_utils.h"
#include <stdlib.h>
#include "arm_math.h"

void arm_add_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL)

#if defined (ARM_MATH_DSP)
  q31_t inA1, inA2;
  q31_t inB1, inB2;
#endif

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A + B */

#if defined (ARM_MATH_DSP)
    /* read 2 times 2 samples at a time from sourceA */
    inA1 = read_q15x2_ia ((q15_t **) &pSrcA);
    inA2 = read_q15x2_ia ((q15_t **) &pSrcA);
    /* read 2 times 2 samples at a time from sourceB */
    inB1 = read_q15x2_ia ((q15_t **) &pSrcB);
    inB2 = read_q15x2_ia ((q15_t **) &pSrcB);

    /* Add and store 2 times 2 samples at a time */
    write_q15x2_ia (&pDst, __QADD16(inA1, inB1));
    write_q15x2_ia (&pDst, __QADD16(inA2, inB2));
#else
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ + *pSrcB++), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ + *pSrcB++), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ + *pSrcB++), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ + *pSrcB++), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A + B */

    /* Add and store result in destination buffer. */
#if defined (ARM_MATH_DSP)
    *pDst++ = (q15_t) __QADD16(*pSrcA++, *pSrcB++);
#else
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ + *pSrcB++), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }
}

void arm_mult_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL)

#if defined (ARM_MATH_DSP)
  q31_t inA1, inA2, inB1, inB2;                  /* Temporary input variables */
  q15_t out1, out2, out3, out4;                  /* Temporary output variables */
  q31_t mul1, mul2, mul3, mul4;                  /* Temporary variables */
#endif

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A * B */

#if defined (ARM_MATH_DSP)
    /* read 2 samples at a time from sourceA */
    inA1 = read_q15x2_ia ((q15_t **) &pSrcA);
    /* read 2 samples at a time from sourceB */
    inB1 = read_q15x2_ia ((q15_t **) &pSrcB);
    /* read 2 samples at a time from sourceA */
    inA2 = read_q15x2_ia ((q15_t **) &pSrcA);
    /* read 2 samples at a time from sourceB */
    inB2 = read_q15x2_ia ((q15_t **) &pSrcB);

    /* multiply mul = sourceA * sourceB */
    mul1 = (q31_t) ((q15_t) (inA1 >> 16) * (q15_t) (inB1 >> 16));
    mul2 = (q31_t) ((q15_t) (inA1      ) * (q15_t) (inB1      ));
    mul3 = (q31_t) ((q15_t) (inA2 >> 16) * (q15_t) (inB2 >> 16));
    mul4 = (q31_t) ((q15_t) (inA2      ) * (q15_t) (inB2      ));

    /* saturate result to 16 bit */
    out1 = (q15_t) __SSAT(mul1 >> 15, 16);
    out2 = (q15_t) __SSAT(mul2 >> 15, 16);
    out3 = (q15_t) __SSAT(mul3 >> 15, 16);
    out4 = (q15_t) __SSAT(mul4 >> 15, 16);

    /* store result to destination */
#ifndef ARM_MATH_BIG_ENDIAN
    write_q15x2_ia (&pDst, __PKHBT(out2, out1, 16));
    write_q15x2_ia (&pDst, __PKHBT(out4, out3, 16));
#else
    write_q15x2_ia (&pDst, __PKHBT(out1, out2, 16));
    write_q15x2_ia (&pDst, __PKHBT(out3, out4, 16));
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

#else
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A * B */

    /* Multiply inputs and store result in destination buffer. */
    *pDst++ = (q15_t) __SSAT((((q31_t) (*pSrcA++) * (*pSrcB++)) >> 15), 16);

    /* Decrement loop counter */
    blkCnt--;
  }
}

void arm_negate_q15(
  const q15_t * pSrc,
        q15_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */
        q15_t in;                                      /* Temporary input variable */

#if defined (ARM_MATH_LOOPUNROLL)

#if defined (ARM_MATH_DSP)
  q31_t in1;                                    /* Temporary input variables */
#endif

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = -A */

#if defined (ARM_MATH_DSP)
    /* Negate and store result in destination buffer (2 samples at a time). */
    in1 = read_q15x2_ia ((q15_t **) &pSrc);
    write_q15x2_ia (&pDst, __QSUB16(0, in1));

    in1 = read_q15x2_ia ((q15_t **) &pSrc);
    write_q15x2_ia (&pDst, __QSUB16(0, in1));
#else
    in = *pSrc++;
    *pDst++ = (in == (q15_t) 0x8000) ? (q15_t) 0x7fff : -in;

    in = *pSrc++;
    *pDst++ = (in == (q15_t) 0x8000) ? (q15_t) 0x7fff : -in;

    in = *pSrc++;
    *pDst++ = (in == (q15_t) 0x8000) ? (q15_t) 0x7fff : -in;

    in = *pSrc++;
    *pDst++ = (in == (q15_t) 0x8000) ? (q15_t) 0x7fff : -in;
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = -A */

    /* Negate and store result in destination buffer. */
    in = *pSrc++;
    *pDst++ = (in == (q15_t) 0x8000) ? (q15_t) 0x7fff : -in;

    /* Decrement loop counter */
    blkCnt--;
  }
}

void arm_offset_q15(
  const q15_t * pSrc,
        q15_t offset,
        q15_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL)

#if defined (ARM_MATH_DSP)
  q31_t offset_packed;                           /* Offset packed to 32 bit */

  /* Offset is packed to 32 bit in order to use SIMD32 for addition */
  offset_packed = __PKHBT(offset, offset, 16);
#endif

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A + offset */

#if defined (ARM_MATH_DSP)
    /* Add offset and store result in destination buffer (2 samples at a time). */
    write_q15x2_ia (&pDst, __QADD16(read_q15x2_ia ((q15_t **) &pSrc), offset_packed));
    write_q15x2_ia (&pDst, __QADD16(read_q15x2_ia ((q15_t **) &pSrc), offset_packed));
#else
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrc++ + offset), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrc++ + offset), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrc++ + offset), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrc++ + offset), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A + offset */

    /* Add offset and store result in destination buffer. */
#if defined (ARM_MATH_DSP)
    *pDst++ = (q15_t) __QADD16(*pSrc++, offset);
#else
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrc++ + offset), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }
}

void arm_scale_q15(
  const q15_t *pSrc,
        q15_t scaleFract,
        int8_t shift,
        q15_t *pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */
        int8_t kShift = 15 - shift;                    /* Shift to apply after scaling */

#if defined (ARM_MATH_LOOPUNROLL)
#if defined (ARM_MATH_DSP)
  q31_t inA1, inA2;
  q31_t out1, out2, out3, out4;                  /* Temporary output variables */
  q15_t in1, in2, in3, in4;                      /* Temporary input variables */
#endif
#endif

#if defined (ARM_MATH_LOOPUNROLL)

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A * scale */

#if defined (ARM_MATH_DSP)
    /* read 2 times 2 samples at a time from source */
    inA1 = read_q15x2_ia ((q15_t **) &pSrc);
    inA2 = read_q15x2_ia ((q15_t **) &pSrc);

    /* Scale inputs and store result in temporary variables
     * in single cycle by packing the outputs */
    out1 = (q31_t) ((q15_t) (inA1 >> 16) * scaleFract);
    out2 = (q31_t) ((q15_t) (inA1      ) * scaleFract);
    out3 = (q31_t) ((q15_t) (inA2 >> 16) * scaleFract);
    out4 = (q31_t) ((q15_t) (inA2      ) * scaleFract);

    /* apply shifting */
    out1 = out1 >> kShift;
    out2 = out2 >> kShift;
    out3 = out3 >> kShift;
    out4 = out4 >> kShift;

    /* saturate the output */
    in1 = (q15_t) (__SSAT(out1, 16));
    in2 = (q15_t) (__SSAT(out2, 16));
    in3 = (q15_t) (__SSAT(out3, 16));
    in4 = (q15_t) (__SSAT(out4, 16));

    /* store result to destination */
    write_q15x2_ia (&pDst, __PKHBT(in2, in1, 16));
    write_q15x2_ia (&pDst, __PKHBT(in4, in3, 16));
#else
    *pDst++ = (q15_t) (__SSAT(((q31_t) *pSrc++ * scaleFract) >> kShift, 16));
    *pDst++ = (q15_t) (__SSAT(((q31_t) *pSrc++ * scaleFract) >> kShift, 16));
    *pDst++ = (q15_t) (__SSAT(((q31_t) *pSrc++ * scaleFract) >> kShift, 16));
    *pDst++ = (q15_t) (__SSAT(((q31_t) *pSrc++ * scaleFract) >> kShift, 16));
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A * scale */

    /* Scale input and store result in destination buffer. */
    *pDst++ = (q15_t) (__SSAT(((q31_t) *pSrc++ * scaleFract) >> kShift, 16));

    /* Decrement loop counter */
    blkCnt--;
  }
}

void arm_shift_q15(
  const q15_t * pSrc,
        int8_t shiftBits,
        q15_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */
        uint8_t sign = (shiftBits & 0x80);             /* Sign of shiftBits */

#if defined (ARM_MATH_LOOPUNROLL)

#if defined (ARM_MATH_DSP)
  q15_t in1, in2;                                /* Temporary input variables */
#endif

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  /* If the shift value is positive then do right shift else left shift */
  if (sign == 0U)
  {
    while (blkCnt > 0U)
    {
      /* C = A << shiftBits */

#if defined (ARM_MATH_DSP)
      /* read 2 samples from source */
      in1 = *pSrc++;
      in2 = *pSrc++;

      /* Shift the inputs and then store the results in the destination buffer. */
#ifndef ARM_MATH_BIG_ENDIAN
      write_q15x2_ia (&pDst, __PKHBT(__SSAT((in1 << shiftBits), 16),
                                     __SSAT((in2 << shiftBits), 16), 16));
#else
      write_q15x2_ia (&pDst, __PKHBT(__SSAT((in2 << shiftBits), 16),
                                      __SSAT((in1 << shiftBits), 16), 16));
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

      /* read 2 samples from source */
      in1 = *pSrc++;
      in2 = *pSrc++;

#ifndef ARM_MATH_BIG_ENDIAN
      write_q15x2_ia (&pDst, __PKHBT(__SSAT((in1 << shiftBits), 16),
                                     __SSAT((in2 << shiftBits), 16), 16));
#else
      write_q15x2_ia (&pDst, __PKHBT(__SSAT((in2 << shiftBits), 16),
                                     __SSAT((in1 << shiftBits), 16), 16));
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

#else
      *pDst++ = __SSAT(((q31_t) *pSrc++ << shiftBits), 16);
      *pDst++ = __SSAT(((q31_t) *pSrc++ << shiftBits), 16);
      *pDst++ = __SSAT(((q31_t) *pSrc++ << shiftBits), 16);
      *pDst++ = __SSAT(((q31_t) *pSrc++ << shiftBits), 16);
#endif

      /* Decrement loop counter */
      blkCnt--;
    }
  }
  else
  {
    while (blkCnt > 0U)
    {
      /* C = A >> shiftBits */

#if defined (ARM_MATH_DSP)
      /* read 2 samples from source */
      in1 = *pSrc++;
      in2 = *pSrc++;

      /* Shift the inputs and then store the results in the destination buffer. */
#ifndef ARM_MATH_BIG_ENDIAN
      write_q15x2_ia (&pDst, __PKHBT((in1 >> -shiftBits),
                                     (in2 >> -shiftBits), 16));
#else
      write_q15x2_ia (&pDst, __PKHBT((in2 >> -shiftBits),
                                     (in1 >> -shiftBits), 16));
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

      /* read 2 samples from source */
      in1 = *pSrc++;
      in2 = *pSrc++;

#ifndef ARM_MATH_BIG_ENDIAN
      write_q15x2_ia (&pDst, __PKHBT((in1 >> -shiftBits),
                                     (in2 >> -shiftBits), 16));
#else
      write_q15x2_ia (&pDst, __PKHBT((in2 >> -shiftBits),
                                     (in1 >> -shiftBits), 16));
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

#else
      *pDst++ = (*pSrc++ >> -shiftBits);
      *pDst++ = (*pSrc++ >> -shiftBits);
      *pDst++ = (*pSrc++ >> -shiftBits);
      *pDst++ = (*pSrc++ >> -shiftBits);
#endif

      /* Decrement loop counter */
      blkCnt--;
    }
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  /* If the shift value is positive then do right shift else left shift */
  if (sign == 0U)
  {
    while (blkCnt > 0U)
    {
      /* C = A << shiftBits */

      /* Shift input and store result in destination buffer. */
      *pDst++ = __SSAT(((q31_t) *pSrc++ << shiftBits), 16);

      /* Decrement loop counter */
      blkCnt--;
    }
  }
  else
  {
    while (blkCnt > 0U)
    {
      /* C = A >> shiftBits */

      /* Shift input and store result in destination buffer. */
      *pDst++ = (*pSrc++ >> -shiftBits);

      /* Decrement loop counter */
      blkCnt--;
    }
  }
}

void arm_sub_q15(
  const q15_t * pSrcA,
  const q15_t * pSrcB,
        q15_t * pDst,
        uint32_t blockSize)
{
        uint32_t blkCnt;                               /* Loop counter */

#if defined (ARM_MATH_LOOPUNROLL)

#if defined (ARM_MATH_DSP)
  q31_t inA1, inA2;
  q31_t inB1, inB2;
#endif

  /* Loop unrolling: Compute 4 outputs at a time */
  blkCnt = blockSize >> 2U;

  while (blkCnt > 0U)
  {
    /* C = A - B */

#if defined (ARM_MATH_DSP)
    /* read 2 times 2 samples at a time from sourceA */
    inA1 = read_q15x2_ia ((q15_t **) &pSrcA);
    inA2 = read_q15x2_ia ((q15_t **) &pSrcA);
    /* read 2 times 2 samples at a time from sourceB */
    inB1 = read_q15x2_ia ((q15_t **) &pSrcB);
    inB2 = read_q15x2_ia ((q15_t **) &pSrcB);

    /* Subtract and store 2 times 2 samples at a time */
    write_q15x2_ia (&pDst, __QSUB16(inA1, inB1));
    write_q15x2_ia (&pDst, __QSUB16(inA2, inB2));
#else
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ - *pSrcB++), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ - *pSrcB++), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ - *pSrcB++), 16);
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ - *pSrcB++), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }

  /* Loop unrolling: Compute remaining outputs */
  blkCnt = blockSize % 0x4U;

#else

  /* Initialize blkCnt with number of samples */
  blkCnt = blockSize;

#endif /* #if defined (ARM_MATH_LOOPUNROLL) */

  while (blkCnt > 0U)
  {
    /* C = A - B */

    /* Subtract and store result in destination buffer. */
#if defined (ARM_MATH_DSP)
    *pDst++ = (q15_t) __QSUB16(*pSrcA++, *pSrcB++);
#else
    *pDst++ = (q15_t) __SSAT(((q31_t) *pSrcA++ - *pSrcB++), 16);
#endif

    /* Decrement loop counter */
    blkCnt--;
  }
}

void arm_mat_init_q15(
  arm_matrix_instance_q15 * S,
  uint16_t nRows,
  uint16_t nColumns,
  q15_t * pData)
{
  /* Assign Number of Rows */
  S->numRows = nRows;

  /* Assign Number of Columns */
  S->numCols = nColumns;

  /* Assign Data pointer */
  S->pData = pData;
}

arm_status arm_mat_mult_q15(
  const arm_matrix_instance_q15 * pSrcA,
  const arm_matrix_instance_q15 * pSrcB,
        arm_matrix_instance_q15 * pDst,
        q15_t                   * pState)
{
        q63_t sum;                                     /* Accumulator */

#if defined (ARM_MATH_DSP)                             /* != CM0 */

        q15_t *pSrcBT = pState;                        /* Input data matrix pointer for transpose */
        q15_t *pInA = pSrcA->pData;                    /* Input data matrix pointer A of Q15 type */
        q15_t *pInB = pSrcB->pData;                    /* Input data matrix pointer B of Q15 type */
        q15_t *px;                                     /* Temporary output data matrix pointer */
        uint16_t numRowsA = pSrcA->numRows;            /* Number of rows of input matrix A */
        uint16_t numColsB = pSrcB->numCols;            /* Number of columns of input matrix B */
        uint16_t numColsA = pSrcA->numCols;            /* Number of columns of input matrix A */
        uint16_t numRowsB = pSrcB->numRows;            /* Number of rows of input matrix A */
        uint32_t col, i = 0U, row = numRowsB, colCnt;  /* Loop counters */
        arm_status status;                             /* Status of matrix multiplication */
        
        q31_t in;                                      /* Temporary variable to hold the input value */
        q31_t inA1, inB1, inA2, inB2;

#ifdef ARM_MATH_MATRIX_CHECK

  /* Check for matrix mismatch condition */
  if ((pSrcA->numCols != pSrcB->numRows) ||
      (pSrcA->numRows != pDst->numRows)  ||
      (pSrcB->numCols != pDst->numCols)    )
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else

#endif /* #ifdef ARM_MATH_MATRIX_CHECK */

  {
    /* Matrix transpose */
    do
    {
      /* The pointer px is set to starting address of column being processed */
      px = pSrcBT + i;

      /* Apply loop unrolling and exchange columns with row elements */
      col = numColsB >> 2U;

      /* First part of the processing with loop unrolling.  Compute 4 outputs at a time.
       ** a second loop below computes the remaining 1 to 3 samples. */
      while (col > 0U)
      {
        /* Read two elements from row */
        in = read_q15x2_ia ((q15_t **) &pInB);

        /* Unpack and store one element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) in;
#else
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

        /* Update pointer px to point to next row of transposed matrix */
        px += numRowsB;

        /* Unpack and store second element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#else
        *px = (q15_t) in;
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */

        /* Update pointer px to point to next row of transposed matrix */
        px += numRowsB;

        /* Read two elements from row */
        in = read_q15x2_ia ((q15_t **) &pInB);

        /* Unpack and store one element in destination */
#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) in;
#else
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */
        px += numRowsB;

#ifndef ARM_MATH_BIG_ENDIAN
        *px = (q15_t) ((in & (q31_t) 0xffff0000) >> 16);
#else
        *px = (q15_t) in;
#endif /* #ifndef ARM_MATH_BIG_ENDIAN */
        px += numRowsB;

        /* Decrement column loop counter */
        col--;
      }

      /* If the columns of pSrcB is not a multiple of 4, compute any remaining output samples here.
       ** No loop unrolling is used. */
      col = numColsB % 0x4U;

      while (col > 0U)
      {
        /* Read and store input element in destination */
        *px = *pInB++;

        /* Update pointer px to point to next row of transposed matrix */
        px += numRowsB;

        /* Decrement column loop counter */
        col--;
      }

      i++;

      /* Decrement row loop counter */
      row--;

    } while (row > 0U);

    /* Reset variables for usage in following multiplication process */
    row = numRowsA;
    i = 0U;
    px = pDst->pData;

    /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    do
    {
      /* For every row wise process, column loop counter is to be initiated */
      col = numColsB;

      /* For every row wise process, pIn2 pointer is set to starting address of transposed pSrcB data */
      pInB = pSrcBT;

      /* column loop */
      do
      {
        /* Set variable sum, that acts as accumulator, to zero */
        sum = 0;

        /* Initiate pointer pInA to point to starting address of column being processed */
        pInA = pSrcA->pData + i;

        /* Apply loop unrolling and compute 2 MACs simultaneously. */
        colCnt = numColsA >> 2U;

        /* matrix multiplication */
        while (colCnt > 0U)
        {
          /* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */

          /* read real and imag values from pSrcA and pSrcB buffer */
          inA1 = read_q15x2_ia ((q15_t **) &pInA);
          inB1 = read_q15x2_ia ((q15_t **) &pInB);

          inA2 = read_q15x2_ia ((q15_t **) &pInA);
          inB2 = read_q15x2_ia ((q15_t **) &pInB);

          /* Multiply and Accumlates */
          sum = __SMLALD(inA1, inB1, sum);
          sum = __SMLALD(inA2, inB2, sum);

          /* Decrement loop counter */
          colCnt--;
        }

        /* process remaining column samples */
        colCnt = numColsA % 0x4U;

        while (colCnt > 0U)
        {
          /* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */
          sum += *pInA++ * *pInB++;

          /* Decrement loop counter */
          colCnt--;
        }

        /* Saturate and store result in destination buffer */
        *px = (q15_t) (__SSAT((sum >> 15), 16));
        px++;

        /* Decrement column loop counter */
        col--;

      } while (col > 0U);

      i = i + numColsA;

      /* Decrement row loop counter */
      row--;

    } while (row > 0U);

#else /* #if defined (ARM_MATH_DSP) */

        q15_t *pIn1 = pSrcA->pData;                    /* Input data matrix pointer A */
        q15_t *pIn2 = pSrcB->pData;                    /* Input data matrix pointer B */
        q15_t *pInA = pSrcA->pData;                    /* Input data matrix pointer A of Q15 type */
        q15_t *pInB = pSrcB->pData;                    /* Input data matrix pointer B of Q15 type */
        q15_t *pOut = pDst->pData;                     /* Output data matrix pointer */
        q15_t *px;                                     /* Temporary output data matrix pointer */
        uint16_t numColsB = pSrcB->numCols;            /* Number of columns of input matrix B */
        uint16_t numColsA = pSrcA->numCols;            /* Number of columns of input matrix A */
        uint16_t numRowsA = pSrcA->numRows;            /* Number of rows of input matrix A    */
        uint32_t col, i = 0U, row = numRowsA, colCnt;  /* Loop counters */
        arm_status status;                             /* Status of matrix multiplication */
        (void)pState;

  /* Check for matrix mismatch condition */
  if ((pSrcA->numCols != pSrcB->numRows) ||
      (pSrcA->numRows != pDst->numRows)  ||
      (pSrcB->numCols != pDst->numCols)    )
  {
    /* Set status as ARM_MATH_SIZE_MISMATCH */
    status = ARM_MATH_SIZE_MISMATCH;
  }
  else

  {
    /* The following loop performs the dot-product of each row in pSrcA with each column in pSrcB */
    /* row loop */
    do
    {
      /* Output pointer is set to starting address of the row being processed */
      px = pOut + i;

      /* For every row wise process, column loop counter is to be initiated */
      col = numColsB;

      /* For every row wise process, pIn2 pointer is set to starting address of pSrcB data */
      pIn2 = pSrcB->pData;

      /* column loop */
      do
      {
        /* Set the variable sum, that acts as accumulator, to zero */
        sum = 0;

        /* Initiate pointer pIn1 to point to starting address of pSrcA */
        pIn1 = pInA;

        /* Matrix A columns number of MAC operations are to be performed */
        colCnt = numColsA;

        /* matrix multiplication */
        while (colCnt > 0U)
        {
          /* c(m,n) = a(1,1) * b(1,1) + a(1,2) * b(2,1) + .... + a(m,p) * b(p,n) */

          /* Perform multiply-accumulates */
          sum += (q31_t) * pIn1++ * *pIn2;
          pIn2 += numColsB;

          /* Decrement loop counter */
          colCnt--;
        }

        /* Convert result from 34.30 to 1.15 format and store saturated value in destination buffer */

        /* Saturate and store result in destination buffer */
        *px++ = (q15_t) __SSAT((sum >> 15), 16);

        /* Decrement column loop counter */
        col--;

        /* Update pointer pIn2 to point to starting address of next column */
        pIn2 = pInB + (numColsB - col);

      } while (col > 0U);

      /* Update pointer pSrcA to point to starting address of next row */
      i = i + numColsB;
      pInA = pInA + numColsA;

      /* Decrement row loop counter */
      row--;

    } while (row > 0U);

#endif /* #if defined (ARM_MATH_DSP) */

    /* Set status as ARM_MATH_SUCCESS */
    status = ARM_MATH_SUCCESS;
  }

  /* Return to application */
  return (status);
}

void v_q_treesum(INTM_T* const vec, ITER_T len, SCALE_T H1, SCALE_T H2) {
  ITER_T count = len, depth = 0;
  int divbytwo = 1;

  while (depth < (H1 + H2)) {
    if (depth >= H1) {
      divbytwo = 0;
    }

    for (ITER_T p = 0; p < ((len >> 1) + 1); p++) {
      if (p < (count >> 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1) + (vec[(2 * p) + 1] >> 1);
          #else
            vec[p] = vec[2 * p] / 2 + vec[(2 * p) + 1] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p] + vec[(2 * p) + 1];
        }
      } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
        if (divbytwo == 1) {
          #ifdef SHIFT
            vec[p] = (vec[2 * p] >> 1);
          #else
            vec[p] = vec[2 * p] / 2;
          #endif
        } else {
          vec[p] = vec[2 * p];
        }
      } else {
        vec[p] = 0;
      }
    }
    count = (count + 1) >> 1;
    depth++;
  }
}

void v_q_add(const INT_T* vec1, const INT_T* vec2, ITER_T len,
             INT_T* ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef CMSISDSP
    INT_T *ret2 = malloc(len * sizeof(INT_T));
    arm_shift_q15(vec1, -(scvec1 + scret), ret, len);
    arm_shift_q15(vec2, -(scvec2 + scret), ret2, len);
    arm_add_q15(ret, ret2, ret, len);
    free(ret2);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec1[i] >> (scvec1 + scret)) + (vec2[i] >> (scvec2 + scret)));
      #else
        ret[i] = ((vec1[i] / scvec1) / scret) + ((vec2[i] / scvec2) / scret);
      #endif
    }
  #endif
}

void v_q_sub(const INT_T* vec1, const INT_T* vec2, ITER_T len,
             INT_T* ret, SCALE_T scvec1, SCALE_T scvec2, SCALE_T scret) {
  #ifdef CMSISDSP
    INT_T *ret2 = malloc(len * sizeof(INT_T));
    arm_shift_q15(vec1, -(scvec1 + scret), ret, len);
    arm_shift_q15(vec2, -(scvec2 + scret), ret2, len);
    arm_sub_q15(ret, ret2, ret, len);
    free(ret2);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec1[i] >> (scvec1 + scret)) - (vec2[i] >> (scvec2 + scret)));
      #else
        ret[i] = ((vec1[i] / scvec1) / scret) - ((vec2[i] / scvec2) / scret);
      #endif
    }
  #endif
}

void v_q_hadamard(const INT_T* vec1, const INT_T* vec2, ITER_T len,
                  INT_T* ret, SCALE_T scvec1, SCALE_T scvec2) {
  #ifdef CMSISDSP
    arm_mult_q15(vec1, vec2, ret, len);
    arm_shift_q15(ret, 15 - (scvec1 + scvec2), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((INTM_T)vec1[i] * (INTM_T)vec2[i]) >> (scvec1 + scvec2);
      #else
        ret[i] = ((((INTM_T)vec1[i] * (INTM_T)vec2[i]) / scvec1) / scvec2);
      #endif
    }
  #endif
}

void v_q_sigmoid(const INT_T* const vec, ITER_T len, INT_T* const ret, INT_T div,
                 INT_T add, INT_T sigmoid_limit, SCALE_T scale_in,
                 SCALE_T scale_out) {
  for (ITER_T i = 0; i < len; i++) {
    INT_T x = (vec[i] / div) + add;

    if (x >= sigmoid_limit) {
      ret[i] = sigmoid_limit << (scale_out - scale_in);
    } else if (x <= 0) {
      ret[i] = 0;
    } else {
      ret[i] = x << (scale_out - scale_in);
    }
  }
}

void v_q_tanh(const INT_T* const vec, ITER_T len, INT_T* const ret,
              SCALE_T scale_in, SCALE_T scale_out) {
  INT_T scale = (1 << scale_in);
  for (ITER_T i = 0; i < len; i++) {
    if (vec[i] >= scale) {
      ret[i] = scale;
    } else if (vec[i] <= -scale) {
      ret[i] = (-scale);
    } else {
      ret[i] = vec[i];
    }
    ret[i] <<= (scale_out - scale_in);
  }
}

void v_q_scalar_add(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_offset_q15(ret, (scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((scalar >> (scscalar + scret)) + (vec[i] >> (scvec + scret)));
      #else
        ret[i] = ((scalar / scscalar) / scret) + ((vec[i] / scvec) / scret);
      #endif
    }
  #endif
}

void v_q_scalar_sub(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_negate_q15(ret, ret, len);
    arm_offset_q15(ret, (scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((scalar >> (scscalar + scret)) - (vec[i] >> (scvec + scret)));
      #else
        ret[i] = ((scalar / scscalar) / scret) - ((vec[i] / scvec) / scret);
      #endif
    }
  #endif
}

void v_q_sub_scalar(const INT_T* vec, INT_T scalar, ITER_T len,
                    INT_T* ret, SCALE_T scvec, SCALE_T scscalar, SCALE_T scret) {
  #ifdef CMSISDSP
    arm_shift_q15(vec, -(scvec + scret), ret, len);
    arm_offset_q15(ret, -(scalar >> (scscalar + scret)), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((vec[i] >> (scvec + scret)) - (scalar >> (scscalar + scret)));
      #else
        ret[i] = ((vec[i] / scvec) / scret) - ((scalar / scscalar) / scret);
      #endif
    }
  #endif
}

void v_q_scalar_mul(INT_T scalar, const INT_T* vec, ITER_T len,
                    INT_T* ret, SCALE_T scscalar, SCALE_T scvec) {
  #ifdef CMSISDSP
    arm_scale_q15(vec, scalar, 15 - (scscalar + scvec), ret, len);
  #else
    for (ITER_T i = 0; i < len; i++) {
      #ifdef SHIFT
        ret[i] = ((INTM_T)scalar * (INTM_T)vec[i]) >> (scscalar + scvec);
      #else
        ret[i] = ((((INTM_T)scalar * (INTM_T)vec[i]) / scscalar) / scvec);
      #endif
    }
  #endif
}

void v_q_argmax(const INT_T* const vec, ITER_T len, ITER_T* const ret) {
  INT_T max_value = vec[0];
  ITER_T max_index = 0;

  for (ITER_T i = 1; i < len; i++) {
    if (max_value < vec[i]) {
      max_index = i;
      max_value = vec[i];
    }
  }

  *ret = max_index;
}

void v_q_relu(INT_T* const vec, ITER_T len) {
  for (ITER_T i = 0; i < len; i++) {
    if (vec[i] < 0) {
      vec[i] = 0;
    }
  }
}

void v_q_exp(const INT_T* const vec, ITER_T len, INT_T* const ret,
             SCALE_T scvec, SCALE_T scret) {
  for (ITER_T i = 0; i < len; i++) {
    ret[i] = ((INT_T)(exp(((float)vec[i]) / scvec) * scret));
  }
}

void v_q_scale_up(INT_T* const vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      vec[i] <<= scvec;
    #else
      vec[i] *= scvec;
    #endif
  }
}

void v_q_scale_down(INT_T* const vec, ITER_T len, SCALE_T scvec) {
  for (ITER_T i = 0; i < len; i++) {
    #ifdef SHIFT
      vec[i] >>= scvec;
    #else
      vec[i] /= scvec;
    #endif
  }
}

void m_q_transpose(const INT_T* const mat, ITER_T nrows, ITER_T ncols,
                   INT_T* const ret) {
  ITER_T len = nrows * ncols, counter = 0;
  for (ITER_T i = 0; i < len; i++) {
    if (counter >= len) {
      counter -= len - 1;
    }

    ret[i] = mat[counter];
    counter += nrows;
  }
}

void m_q_reverse(const INT_T* const mat, ITER_T nrows, ITER_T ncols, ITER_T axis,
                 INT_T* const ret) {
  ITER_T len = nrows * ncols;

  if (axis == 0) {
    ITER_T col_counter = 0, row_index = len - ncols;

    for (ITER_T i = 0; i < len; i++) {
      if (col_counter >= ncols) {
        col_counter = 0;
        row_index -= ncols;
      }

      ret[i] = mat[row_index + col_counter];
      col_counter++;
    }
  } else {
    S_ITER_T row_counter = ncols - 1;
    ITER_T col_index = 0;

    for (ITER_T i = 0; i < len; i++) {
      if (row_counter < 0) {
        row_counter = ncols - 1;
        col_index += ncols;
      }

      ret[i] = mat[col_index + (ITER_T)row_counter];
      row_counter--;
    }
  }
}

void m_q_add_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nrows, ITER_T ncols, INT_T* const ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nrows * ncols;
  for (ITER_T i = 0, w = 0; i < len; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) + (vec[w] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) + ((vec[w] / scvec) / scret);
    #endif
  }
}

void m_q_sub_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nrows, ITER_T ncols, INT_T* const ret,
                 SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nrows * ncols;
  for (ITER_T i = 0, w = 0; i < len; i++, w++) {
    if (w >= ncols) {
      w = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) - (vec[w] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) - ((vec[w] / scvec) / scret);
    #endif
  }
}

void m_q_mulvec(const INT_T* mat, const INT_T* vec, ITER_T nrows,
                ITER_T ncols, INT_T* ret, SCALE_T scmat, SCALE_T scvec,
                SCALE_T H1, SCALE_T H2) {
  #ifdef CMSISDSP
    INT_T *tmp = malloc(ncols * sizeof(INT_T));
    INT_T *ret2 = malloc(ncols * nrows * sizeof(INT_T));
    INT_T *ret3 = malloc(ncols * sizeof(INT_T));
    arm_shift_q15(mat, -scmat, ret2, nrows * ncols);
    arm_shift_q15(vec, -scvec, ret3, ncols);
    arm_matrix_instance_q15 A, B, C;
    arm_mat_init_q15(&A, nrows, ncols, ret2);
    arm_mat_init_q15(&B, ncols, 1, ret3);
    arm_mat_init_q15(&C, nrows, 1, ret);
    arm_mat_mult_q15(&A, &B, &C, tmp);
    free(tmp);
    free(ret2);
    free(ret3);
  #else
    INTM_T tmp[ncols];
    for (ITER_T row = 0; row < nrows; row++) {
      INT_T* mat_offset = (INT_T*)mat + row * ncols;

      for (ITER_T col = 0; col < ncols; col++) {
        tmp[col] = ((INTM_T)(*mat_offset++) * (INTM_T)vec[col]);
      }

      v_q_treesum(&tmp[0], ncols, H1, H2);
      #ifdef SHIFT
        ret[row] = (tmp[0] >> (scmat + scvec));
      #else
        ret[row] = ((tmp[0] / scmat) / scvec);
      #endif
    }
  #endif
}

void m_q_sparse_mulvec(const ITER_T* const col_indices, const INT_T* const mat_values,
                       const INT_T* const vec, ITER_T ndims, INT_T* const ret,
                       SCALE_T scmat, SCALE_T scvec, SCALE_T scret) {
  ITER_T iter_index = 0, iter_value = 0;
  for (ITER_T k = 0; k < ndims; k++) {
    ITER_T index = col_indices[iter_index];

    while (index != 0) {
      #ifdef SHIFT
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) >> (scmat + scvec + scret));
      #else
        ret[index - 1] += (((INTM_T)mat_values[iter_value] * (INTM_T)vec[k]) / ((INTM_T)scmat * (INTM_T)scvec * (INTM_T)scret));
      #endif
      iter_index++;
      iter_value++;
      index = col_indices[iter_index];
    }

    iter_index++;
  }
}

void t_q_add_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                 ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                 SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) + (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) + ((vec[c] / scvec) / scret);
    #endif
  }
}

void t_q_sub_vec(const INT_T* const mat, const INT_T* const vec,
                 ITER_T nbatches, ITER_T nrows, ITER_T ncols,
                 ITER_T nchannels, INT_T* const ret, SCALE_T scmat,
                 SCALE_T scvec, SCALE_T scret) {
  ITER_T len = nbatches * nrows * ncols * nchannels;
  for (ITER_T i = 0, c = 0; i < len; i++, c++) {
    if (c >= nchannels) {
      c = 0;
    }

    #ifdef SHIFT
      ret[i] = ((mat[i] >> (scmat + scret)) - (vec[c] >> (scvec + scret)));
    #else
      ret[i] = ((mat[i] / scmat) / scret) - ((vec[c] / scvec) / scret);
    #endif
  }
}

void q_maxpool(const INT_T* const input, INT_T* const output, ITER_T N,
               ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF, ITER_T CF,
               ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G, S_ITER_T HPadU,
               S_ITER_T HPadD, S_ITER_T WPadL, S_ITER_T WPadR, ITER_T HStride,
               ITER_T WStride, ITER_T HDilation, ITER_T WDilation,
               SCALE_T scinput, SCALE_T scoutput) {
  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * (S_ITER_T)((HF - 1) >> 1)) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * (S_ITER_T)((WF - 1) >> 1)) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * (S_ITER_T)(HF >> 1)) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * (S_ITER_T)(WF >> 1)) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          for (ITER_T c = 0; c < COut; c++) {

            INT_T max = INT_TMIN;
            for (S_ITER_T hf = -((HF - 1) >> 1); hf <= (HF >> 1); hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              for (S_ITER_T wf = -((WF - 1) >> 1); wf <= (WF >> 1); wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    if (max < 0) {
                      max = 0;
                    }
                  } else {
                    INT_T a = input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)];
                    if (max < a) {
                      max = a;
                    }
                  }
                }
              }
            }

            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (max >> (scinput + scoutput));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = ((max / scinput) / scoutput);
            #endif
          }
        }
      }
    }
  }
}

void q_convolution(const INT_T* const input, const INT_T* const filter,
                   INT_T* const output, INTM_T* const treesumBuffer, ITER_T N,
                   ITER_T H, ITER_T W, ITER_T CIn, ITER_T HF, ITER_T WF,
                   ITER_T CF, ITER_T COut, ITER_T HOut, ITER_T WOut, ITER_T G,
                   S_ITER_T HPadU, S_ITER_T HPadD, S_ITER_T WPadL,
                   S_ITER_T WPadR, ITER_T HStride, ITER_T WStride,
                   ITER_T HDilation, ITER_T WDilation, SCALE_T H1, SCALE_T H2,
                   SCALE_T scinput, SCALE_T scoutput) {
  S_ITER_T HOffsetL = ((S_ITER_T)HDilation * (S_ITER_T)((HF - 1) >> 1)) - HPadU;
  S_ITER_T WOffsetL = ((S_ITER_T)WDilation * (S_ITER_T)((WF - 1) >> 1)) - WPadL;
  S_ITER_T HOffsetR = ((S_ITER_T)HDilation * (S_ITER_T)(HF >> 1)) - HPadD;
  S_ITER_T WOffsetR = ((S_ITER_T)WDilation * (S_ITER_T)(WF >> 1)) - WPadR;

  ITER_T HOffsetIn = W * CIn;
  ITER_T NOffsetIn = H * HOffsetIn;
  ITER_T WOffsetF = CF * COut;
  ITER_T HOffsetF = WF * WOffsetF;
  ITER_T WOffsetOut = (COut * G);
  ITER_T HOffsetOut = WOut * WOffsetOut;
  ITER_T NOffsetOut = HOut * HOffsetOut;
  for (ITER_T n = 0; n < N; n++) {
    ITER_T hout = 0;
    ITER_T NIndexIn = n * NOffsetIn;
    ITER_T NIndexOut = n * NOffsetOut;
    for (S_ITER_T h = HOffsetL; h < (S_ITER_T)H - HOffsetR; h += (S_ITER_T)HStride, hout++) {
      ITER_T wout = 0;
      ITER_T HIndexOut = hout * HOffsetOut;
      for (S_ITER_T w = WOffsetL; w < (S_ITER_T)W - WOffsetR; w += (S_ITER_T)WStride, wout++) {
        ITER_T WIndexOut = wout * WOffsetOut;
        for (ITER_T g = 0; g < G; g++) {
          ITER_T CIndexIn = g * CF;
          ITER_T CIndexOut = g * COut;
          for (ITER_T c = 0; c < COut; c++) {

            ITER_T counter = 0;
            for (S_ITER_T hf = -((HF - 1) >> 1); hf <= (HF >> 1); hf++) {
              S_ITER_T hoffset = h + ((S_ITER_T)HDilation * hf);
              ITER_T HIndexIn = ((ITER_T)hoffset) * HOffsetIn;
              ITER_T HIndexF = ((ITER_T)(hf + ((HF - 1) >> 1))) * HOffsetF;
              for (S_ITER_T wf = -((WF - 1) >> 1); wf <= (WF >> 1); wf++) {
                S_ITER_T woffset = w + ((S_ITER_T)WDilation * wf);
                ITER_T WIndexIn = ((ITER_T)woffset) * CIn;
                ITER_T WIndexF = ((ITER_T)(wf + ((WF - 1) >> 1))) * WOffsetF;
                for (ITER_T cf = 0; cf < CF; cf++) {
                  if ((hoffset < 0) || (hoffset >= (S_ITER_T)H) || (woffset < 0) || (woffset >= (S_ITER_T)W)) {
                    treesumBuffer[counter] = 0;
                  } else {
                    treesumBuffer[counter] = ((INTM_T)input[NIndexIn + HIndexIn + WIndexIn + (cf + CIndexIn)]) *
                      ((INTM_T)filter[HIndexF + WIndexF + (c + cf * COut)]);
                  }
                  counter++;
                }
              }
            }

            v_q_treesum(&treesumBuffer[0], HF * WF * CF, H1, H2);
            #ifdef SHIFT
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = (treesumBuffer[0] >> (scinput + scoutput));
            #else
              output[NIndexOut + HIndexOut + WIndexOut + (c + CIndexOut)] = ((treesumBuffer[0] / scinput) / scoutput);
            #endif
          }
        }
      }
    }
  }
}
