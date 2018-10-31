/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT license.
 *
 * Configuration parameters
 */
#ifndef __CONFIG__
#define __CONFIG__

/*
 * The number of buckets that are in one feature
 * vector
 */
#define NUM_BUCKETS 			20
/*
 * The stride in terms of number of
 * measurements (instances) that is used for featurization.
 */
#define STRIDE					20
/*
 * The width of the internal buffer that retains
 * a subwindow worth of measurements. 2 x because
 * we have to keep track of acceleration and gyro
 * values. (Accel and Gyro values are of type Vector16).
 */
#define BUCKET_BUFF_WIDTH 		(2 * BUCKET_WIDTH)
#define BUCKET_WIDTH 			20
#define FEATURE_LENGTH			(6 * BUCKET_WIDTH + 4) 
/*
 * For the current set of features, we have 6 features
 * per bucket - the standard deviations for the 6 raw
 * values. Hence the feature vector dimension is 
 * `6 x NUM_BUCKETS`+ Indices and length of max and min values of gy.
 */
#define FEAT_VEC_DIM 			((6 * NUM_BUCKETS) + 4)

/* Minimum and Maximum Acceleration and Gyro
 * Values for Min-Max normalisation.
 * Refer ./src/minmaxnormalize.h
 */
#define MIN_ACC 				-16384
#define MAX_ACC 				16384
#define MIN_GYR_X 				-512
#define MIN_GYR_Y 				-2048
#define MIN_GYR_Z				-512
#define MAX_GYR_X 				512
#define MAX_GYR_Y 				2048
#define MAX_GYR_Z 				512

#endif // __CONFIG__