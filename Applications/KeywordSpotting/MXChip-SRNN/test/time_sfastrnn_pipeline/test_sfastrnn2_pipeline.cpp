#include <sfastrnnpipeline.h>
#include <testdata.h>
#include <Arduino.h>

extern struct FastRNNParams fastrnnParams_test0;
extern struct FastRNNParams fastrnnParams_test1;
extern struct FCParams fcParams_test;
extern void initFastRNN_test0();
extern void initFastRNN_test1();
extern void initFC_test();

#ifdef DEBUG_MODE
#define PRECISION 4
void printVoid(void *val){
    int32_t a = *((int32_t*)val);
    Serial.println(a);
}

void printStr(char *a){
    Serial.print(a);
}

void printInt32(int32_t val){
    Serial.println(val);
}

void printFloatAddr(float *a){
    Serial.printf("%p\n", a);
}

void printHexQ31(q31_t val){
    char buff[20];
    sprintf(buff, "%p", *(int*)&val);
   Serial.println(buff);
}

void printFloatArrF32(float32_t *arr, int len, float scale){
    for(int i = 0; i < len; i++){
        float val = ((float*)arr)[i];
        Serial.print(val * 1.0, PRECISION); Serial.print(", ");
    }
    Serial.println();
    delay(1000);
}

void printIntArr(int32_t *arr, int len, int offset){
    for(int i = 0; i < len; i++){
        int32_t val = arr[i + offset];
        Serial.print(val); Serial.print(", ");
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
    Screen.init();
    delay(500);
    initFastRNN_test0();
    initFastRNN_test1();
    initFC_test();
    Serial.println();
    Serial.println("Ready");
    Screen.print(1, "Ready");
    delay(500);
    Screen.clean();
    unsigned ret = sfastrnn2p_init(&fastrnnParams_test0,
        &fastrnnParams_test1, &fcParams_test);
    Serial.printf("Return code: %d (init)\n", ret);
    if(ret != 0)
    error("Shallow FastRNN initialization failed (code %d)", ret);
    if(ret != 0) while(1);

    int32_t test[11] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
}

void loop(){
    Serial.printf("New Loop\n");
    Screen.print(1, "New Loop");
    unsigned ret = 0;
    // Push 160 samples every 10ms
    for(int i = 0; i < 100; i++){
        ret = sfastrnn2p_add_new_samples(&test_audio[i * 160], 160);
        ret = ret | ret;
        wait_ms(10);
    }
    // Wait for the consumer to finish
    delay(1000);
    Serial.printf("Return code: %d (push)\n", ret);
    Screen.clean();
    delay(500);
}

int main(){
    setup();
    delay(500);
    Serial.println("NEW");
    delay(500);
    for(int i = 0; i < 25; i++)
        loop();
    sfastrnn2p_quit();
    delay(500);
    Screen.print("Done");
}

