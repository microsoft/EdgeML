/*
 * Prediction Pipeline with Shallow FastRNN
 * ----------------------------------------
 * 
 * This is written in CPP for mbed-os
 * 
 * Currently supports 2 layer shallow fastRNN fed features
 * through a 512-point FFT based log-fbank featurizer. 
 * 
 */
#include <arm_math.h>
#include "../algorithms/sfastrnn.h"
#include "../algorithms/fc.h"
#include "../utils/circularq.h"
#include "../featurizer/logfbank.h"
#include <Thread.h>
#include <Mutex.h>
#include <string.h>
#include <mbed_wait_api.h>
#include <mbed_error.h>
#ifdef DEBUG_MODE
    #include "../debug_mode/debugmethods.h"
#endif

#define TIME_STEPS0 8
#define TIME_STEPS1 6
#define HID_STATES0 16
#define HID_STATES1 16
#define NUM_LABELS  13
// The length of the container that will be used to
// hold the feature vector  (number_filt x timesteps0)
#define FEAT_BUFFER_LEN                           (TIME_STEPS0 * NFILT)
// The length of the container that will be used to
// hold the intermediate hidden sates (hiddenDim0 x timesteps1)
#define H0_BUFFER_LEN                             (TIME_STEPS1 * HID_STATES0)
// Hidden state 0 len
#define H0_LEN                                     HID_STATES0
// The length of the final hidden state. This will be input
// to the FC layer
#define FINAL_H_LEN                               HID_STATES1
// We need a buffer to keep the audio that is pushed
// in through add_new_samples. This buffer is serviced
// by the featurizer, which will flush the buffer once
// featurization is complete.
#define AUDIO_SAMPLES_BUFFER_LEN                   2048
// If feature normalization is required
#ifdef NORMALIZE_FEAT
    extern float featNormMean[];
    extern float featNormStd[];
#endif


// Error Codes
// -----------
#define SFASTRNN2P_SUCCESS                         0
// H0_CONTAINER_LEN is defined to an incorrect value
// This usually means that HID_STATES0, p0->statesLen
// or p1->timeSteps are wrong.
#define SFASTRNN2P_H0_BUFFER_LEN_ERR               1
// The hidden states len of layer 0 and the input len
// of layer 1 does not match.
#define SFASTRNN2P_SFASTRNN2_INIT_ERR              2
#define SFASTRNN2P_AUDIO_SAMPLES_BUFFER_FULL       3
// This happens when either the buffer is full or when
// the queue is corrupted. (check source in circularq.c)
#define SFASTRNN2P_AUDIO_SAMPLES_BUFFER_PUSH_ERR   4
#define SFASTRNN2P_PRED_THR_SPWAN_ERR              5
// NFFT Should be >= FRAME_LEN
#define SFASTRNN2P_NFFT_TOO_SMALL_ERR              6
// FEAT_BUFFER_LEN should be timeSteps0 * featLen0
#define SFASTRNN2P_FEAT_BUFFER_LEN_ERR             7
// FINAL_H_LEN should be statesLen1
#define SFASTRNN2P_FINAL_H_LEN_ERR                 8
// H0_BUFFER_LEN should be statesLen0
#define SFASTRNN2P_H0_LEN_ERR                      9
// We coun't not get a lock on the audio queue to push
// new audio even after waiting for 1 ms. These samples
// will be dropped.
#define SFASTRNN2P_AUDIO_QUEUE_MUTEX_LOCK_ERR      10
// Init function called twice. Since we spawn threads
// internally, we can't support this. A NoOP second init
// can be supported (that which just returns SUCCESS)
// but I don't want to enable this behaviour to enforce
// programmer awareness.
#define SFASTRNN2P_MULTIPLE_INIT_ERROR             11
#define SFASTRNN2P_UNINITIALIZED                   12
// The output len specified by the fc params does
// not match the ones specified in NUM_LABELS
#define SFASTRNN2P_FC_OUT_LEN_ERR                  13
// The output len specified in FINAL_H_LEN does not
// match the input required by fc parameters.
#define SFASTRNN2P_FC_IN_LEN_ERR                   14

// Initialize the model with FastRNNParams and FC params
// This method also starts the prediction thread and waits
// for audio, pushed through using the add_new_samples
// method.
// p0, p1, fc: Model parameters
// prediction_cb: A call_back function invoked when a *non-0* class
//      is prediction. The prediction score vector after the
//      FC layer is passed as arguments along with its length.
unsigned sfastrnn2p_init(struct FastRNNParams *p0,
    struct FastRNNParams *p1, struct FCParams *fc,
    void (*prediction_cb)(float *, int));
// Add new audio samples to the prediction pipeline.
unsigned sfastrnn2p_add_new_samples(int16_t *samples, int len);
// Stop the prediction thread.
void sfastrnn2p_quit();
// Return the state of the prediction thread.
rtos::Thread::State get_thread_state();
