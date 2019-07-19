#include "../sfastrnn_pipeline/sfastrnnpipeline.h"

static unsigned init_error_code = SFASTRNN2P_UNINITIALIZED;
// Will be used as the buffer to hold the layer 0 input.
// That is, the feature vector
static float feat_container[FEAT_BUFFER_LEN];
static float flat_feat_vec[FEAT_BUFFER_LEN];
static float mfcc_result[NFILT];
FIFOCircularQ featVecQ;
// Will be used as the buffer to hold the layer 1 input.
// That is, the layer 0 hidden states.
static float h0_container[H0_BUFFER_LEN];
// Will host the SFastRNNParams for 2 layer network
static SFastRNNParams2 sfastrnn2_params;
static float h0_buffer[H0_LEN];
static float inp1_buffer[H0_BUFFER_LEN];
// Will host the output states of SFastRNN2
static float final_h[FINAL_H_LEN];
// Will host the FC layer and its output from the FC layer
struct FCParams *fcparams;
static float logits[NUM_LABELS];
// A queue, its container and mutex to hold audio samples.
FIFOCircularQ audioQ;
static int16_t audio_samples_container[AUDIO_SAMPLES_BUFFER_LEN];
rtos::Mutex audio_buffer_mutex;
// Prediction function for a separate thread
rtos::Thread pred_thread;
void pred_func();
// One frame of audio
static int32_t audio_frame_buffer[NFFT];
static int8_t quit_prediction_flag;
// We need to keep track of the tail element
// of each audio frame to make sure pre-emphasis
// is correctly performed
static int16_t preemph_tail;
// For F-bank filters
static float32_t fbank_f32[NFILT * (NFFT/ 2 + 1)];
#ifdef FFT_Q31
    static q31_t fbank_q31[NFILT * (NFFT/ 2 + 1)];
#endif
static void (*prediction_cb)(float *, int);


unsigned sfastrnn2p_init(struct FastRNNParams *p0,
    struct FastRNNParams *p1, struct FCParams *fc,
    void (*pred_cb)(float*, int)){
    if(init_error_code != SFASTRNN2P_UNINITIALIZED)
        return SFASTRNN2P_MULTIPLE_INIT_ERROR;

    unsigned ret = 0;
    ret = initSFastRNN2(&sfastrnn2_params, p0, p1, h0_container,
        h0_buffer, inp1_buffer);
    // Happens if input and output dimensions mismatch
    if(ret != 0)
        return SFASTRNN2P_SFASTRNN2_INIT_ERR;
    int size = p0->statesLen * p1->timeSteps;
    if (size != H0_BUFFER_LEN)
        return SFASTRNN2P_H0_BUFFER_LEN_ERR;
    // Check if final_h has correct dimensions
    if (FINAL_H_LEN != p1->statesLen)
        return SFASTRNN2P_FINAL_H_LEN_ERR;
    if (H0_LEN != p0->statesLen)
        return SFASTRNN2P_H0_LEN_ERR;
    // Initialize the FC layer and check for output len correctness
    fcparams = fc;
    if (NUM_LABELS != fcparams->outputDim)
        return SFASTRNN2P_FC_OUT_LEN_ERR;
    if (FINAL_H_LEN != fcparams->inputDim)
        return SFASTRNN2P_FC_IN_LEN_ERR;
    // Initialize the audio queue 
    q_init(&audioQ, audio_samples_container, AUDIO_SAMPLES_BUFFER_LEN,
        cb_write_int16, cb_read_int16);
    // Initialize the feature vector buffer
    size = p0->timeSteps * p0->featLen;
    if (size != FEAT_BUFFER_LEN)
        return SFASTRNN2P_FEAT_BUFFER_LEN_ERR;
    q_init(&featVecQ, feat_container, FEAT_BUFFER_LEN,
        cb_write_float, cb_read_float);
    // Initialize fbanks
    get_filterbank_parameters(fbank_f32, NFILT, SAMPLING_RATE, NFFT);
    #ifdef FFT_Q31
        arm_float_to_q31(fbank_f32, fbank_q31, NFILT * (NFFT/2 + 1));
    #endif
    // Initialize the frame buffer
    if(NFFT < FRAME_LEN)
        return SFASTRNN2P_NFFT_TOO_SMALL_ERR;
    memset(audio_frame_buffer, 0, NFFT * sizeof(int32_t));
    // If we these methods don't function, we don't return.
    // Also, we are royally screwed.
    audio_buffer_mutex.lock();
    audio_buffer_mutex.unlock();
    // Start the prediction thread at default priority
    // To pass arguments, use the callback API in RTOS
    osStatus status = pred_thread.start(pred_func);
    if (status != osOK)
        return SFASTRNN2P_PRED_THR_SPWAN_ERR;
    quit_prediction_flag = 0;
    preemph_tail = 0;
    // Set the prediction call_back
    prediction_cb = pred_cb;
    return SFASTRNN2P_SUCCESS;
}

unsigned sfastrnn2p_add_new_samples(int16_t *samples, int len){
    int size = q_getSize(&audioQ);
    if(AUDIO_SAMPLES_BUFFER_LEN - size < len)
        return SFASTRNN2P_AUDIO_SAMPLES_BUFFER_FULL;
    // Try to lock with timeout of 1 ms. Should be enough.
    int rett = audio_buffer_mutex.lock(1);
    if(rett != osOK)
        return SFASTRNN2P_AUDIO_QUEUE_MUTEX_LOCK_ERR;
    // Don't use force push. We have asserted that there is enough space
    // The only reason for this to fail is a failed buffer. We want to
    // be aware of that.
    int ret = q_enqueue_batch(&audioQ, (void *)samples,
        sizeof(int16_t), len);
    audio_buffer_mutex.unlock();
    if(ret != 0)
        return SFASTRNN2P_AUDIO_SAMPLES_BUFFER_PUSH_ERR;
    return SFASTRNN2P_SUCCESS;
}

void pred_func(){
    int qsize;
    int debug_fftTime = 0;
    while (quit_prediction_flag == 0){
        qsize = q_getSize(&audioQ);
        if(qsize < FRAME_LEN){
            // sleep for 5 ms
            rtos:wait_ms(5);
            continue;
        }
        while(qsize >= FRAME_LEN){
            audio_buffer_mutex.lock();
            qsize = q_getSize(&audioQ);
            for(int i = 0; i < FRAME_LEN; i++)
                audio_frame_buffer[i] = (int32_t)(*(int16_t*)q_atN(&audioQ, i));
            for(int i = 0; i < STRIDE; i++)
                q_del_oldest(&audioQ);
            audio_buffer_mutex.unlock();
            int32_t tailnew = audio_frame_buffer[STRIDE - 1];
            // Compute FFT and push to feature vector buffer
            #ifdef FFT_Q31
            logfbank(mfcc_result, audio_frame_buffer, fbank_q31, preemph_tail);
            #elif FFT_F32
            logfbank(mfcc_result, audio_frame_buffer, fbank_f32, preemph_tail);
            #endif
            // If required perform normalization
            #ifdef NORMALIZE_FEAT
            for(int i = 0; i < NFILT; i++){
                mfcc_result[i] = (mfcc_result[i] - featNormMean[i]);
                mfcc_result[i] /= featNormStd[i];
            }
            #endif
            preemph_tail = tailnew;
            // Feature vector is never full - maintained as an invariant.
            // If its is full, we are not processing fast enough. Fail
            int ret = q_enqueue_batch(&featVecQ, mfcc_result,
                sizeof(float), NFILT);
            if (ret != 0)
                error("PredThrErr: Critial: featVecQ push fail (code %d)", ret);
            // If feature vector buffer is full, flatten and make prediction
            // and empty. Maintain the invariant that feature_vec will allow
            // for at least one push
            if (q_is_full(&featVecQ)){
                q_flatten_float(&featVecQ, flat_feat_vec);
                q_reset(&featVecQ);
                // We have a full feature vector. Make a prediction.
                SFastRNNInference2(&sfastrnn2_params, flat_feat_vec, final_h);
                FCInference(fcparams, final_h, logits, 0);
                prediction_cb(logits, 13);
            }
            qsize = q_getSize(&audioQ);
        }
    }
}

void sfastrnn2p_quit(){
    quit_prediction_flag = 1;
}

rtos::Thread::State get_thread_state(){
    return pred_thread.get_state();
}