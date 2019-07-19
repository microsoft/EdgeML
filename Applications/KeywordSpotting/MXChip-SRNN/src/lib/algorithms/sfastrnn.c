#include "sfastrnn.h"

unsigned initSFastRNN2(struct SFastRNNParams2 *mis2,
        struct FastRNNParams *p0, struct FastRNNParams *p1,
        float *container1, float *h0_buffer, float *inp1_buffer){
    mis2->frnn0 = p0;
    mis2->frnn1 = p1;
    mis2->timeSteps0 = p0->timeSteps;
    mis2->timeSteps1 = p1->timeSteps;
    mis2->__featLen0 = p0->featLen;
    mis2->__featLen1 = p0->statesLen;
    if (mis2->__featLen1 != p1->featLen)
        return 1;
    int size = p1->timeSteps * p0->statesLen;
    q_init(&mis2->h0q, container1, size,
            cb_write_float, cb_read_float);
    mis2->h0_buffer = h0_buffer;
    mis2->inp1_buffer = inp1_buffer;
    return 0;
}

void SFastRNNInference2(struct SFastRNNParams2 *params, const float *x,
        float *result_h) {
    unsigned statesLen0 = params->frnn0->statesLen;
    unsigned timeSteps1 = params->frnn1->timeSteps;
    float *h0 = params->h0_buffer;
    float *inp1 = params->inp1_buffer;
    FastRNNInference(params->frnn0, x, h0);
    q_force_enqueue_batch(&(params->h0q), h0, sizeof(float), statesLen0);
    q_flatten_float(&(params->h0q), inp1);
    FastRNNInference(params->frnn1, inp1, result_h);
}

