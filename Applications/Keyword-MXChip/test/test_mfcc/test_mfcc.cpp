#include <Arduino.h>
#include "logfbank.h"
#include "data.h"

#define PRECISION 4

static float32_t mfccResult[NFILT];
static float32_t fbank_f32[NFILT * (NFFT/ 2 + 1)];
#ifdef FFT_Q31
    static q31_t fbank_q31[NFILT * (NFFT/ 2 + 1)];
#endif

#ifdef DEBUG_MODE
void printVoid(void *val){
    int32_t a = *((int32_t*)val);
    Serial.println(a);
}

void printInt32(int32_t val){
   Serial.println(val);
}

void printHexQ31(q31_t val){
    char buff[20];
    sprintf(buff, "%p", *(int*)&val);
   Serial.println(buff);
}

void printFloatArrF32(float32_t *arr, int len, float scale){
    for(int i = 0; i < len; i++){
        float val = arr[i];
        Serial.print((float)val * scale, PRECISION); Serial.print(", ");
    }
    Serial.println();
    delay(1000);
}

void printFloatArrQ31(q31_t *arr, int len, float scale){
    for(int i = 0; i < len; i++){
        float32_t val;
        arm_q31_to_float(&arr[i], &val, 1);
        Serial.print((float)val * scale, PRECISION); Serial.print(", ");
    }
    Serial.println();
    delay(1000);
}
#endif // DEBUG_MODE

void setup(){
    Serial.begin(115200);
    delay(500);
    Serial.println("Ready");
    delay(500);
    get_filterbank_parameters(fbank_f32, NFILT, SAMPLING_RATE, NFFT);
    #ifdef FFT_Q31
        arm_float_to_q31(fbank_f32, fbank_q31, NFILT * (NFFT/2 + 1));
    #endif
    delay(500);
}


void loop(){
    #ifdef FFT_F32
    void *_fbank = (void *) fbank_f32;
    #elif FFT_Q31
    void *_fbank = (void *) fbank_q31;
    #endif
    Serial.println("New Loop");
    delay(1000);
    unsigned long startTime = micros();
    for(int i = 0; i < 100; i++){
        logfbank(mfccResult, inputData, _fbank, (int32_t)0);
    }
    unsigned long endTime = micros();
    float totalTime = (endTime - startTime) / 1000.0;
    Serial.print("Time (ms) for 100 512 point MFCC is: ");
    Serial.println(totalTime, 2);
    #ifdef DEBUG_MODE
    printFloatArrF32(mfccResult, NFILT, 1);
    #endif
    delay(1000);
    Serial.println();
}

int main(){
    setup();
    for(int i = 0; i < 100; i++){
        loop();
    }
}