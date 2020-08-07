// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#ifndef __QUANTIZED_FACE_DETECTION_H__
#define __QUANTIZED_FACE_DETECTION_H__

#include "quantized_fastgrnn.h"
#include "quantized_rnnpool.h"
#include "quantized_mbconv.h"

#include "q_wider_face_detection_model/conv2D.h"
#include "q_wider_face_detection_model/rnn1.h"
#include "q_wider_face_detection_model/rnn2.h"
#include "q_wider_face_detection_model/mbconv1.h"
#include "q_wider_face_detection_model/mbconv2.h"
#include "q_wider_face_detection_model/mbconv3.h"
#include "q_wider_face_detection_model/mbconv4.h"
#include "q_wider_face_detection_model/detection1.h"
#include "q_wider_face_detection_model/mbconv5.h"
#include "q_wider_face_detection_model/mbconv6.h"
#include "q_wider_face_detection_model/mbconv7.h"
#include "q_wider_face_detection_model/mbconv8.h"
#include "q_wider_face_detection_model/detection2.h"
#include "q_wider_face_detection_model/mbconv9.h"
#include "q_wider_face_detection_model/mbconv10.h"
#include "q_wider_face_detection_model/mbconv11.h"
#include "q_wider_face_detection_model/detection3.h"
#include "q_wider_face_detection_model/mbconv12.h"
#include "q_wider_face_detection_model/mbconv13.h"
#include "q_wider_face_detection_model/mbconv14.h"
#include "q_wider_face_detection_model/detection4.h"
