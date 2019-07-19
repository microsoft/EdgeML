#include <sfastrnnpipeline.h>
#include <model.h>
#include <Arduino.h>
#include <AudioClassV2.h>

extern struct FastRNNParams fastrnnParams0;
extern struct FastRNNParams fastrnnParams1;
extern struct FCParams fcParams;
extern void initFastRNN0();
extern void initFastRNN1();
extern void initFC();
extern const char *labelInvArr[];
// A circular q for voting
#define VOTE_WIN_LEN 10
#define VOTE_MAJORITY 5
FIFOCircularQ votingQ;
static int votingContainer[VOTE_WIN_LEN];
static int votingFrequence[NUM_LABELS];

// TODO: Explain this
#define TRANSFER_BUFFER_MAX_LEN 128
static AudioClass& Audio = AudioClass::getInstance();
char readBuffer[AUDIO_CHUNK_SIZE];
static int transfer_buffer_curr_len = 0;
static int16_t transfer_buffer[TRANSFER_BUFFER_MAX_LEN];

void recordCallback(void) {
    int length = Audio.readFromRecordBuffer(readBuffer, AUDIO_CHUNK_SIZE);
    // We are 16bit (short) and not 8bit (char). Hence actual number of samples
    // is half. Further, we need to ignore the second channel in the
    // audio (interleaved with the first channel).
    length = length / 2;
    length = length - (length % 2);
    length = length / 2;
    if(length > TRANSFER_BUFFER_MAX_LEN)
        error("Transfer buffer too small");
    // Convert to 16 bit samples
    int16_t *tempAudio = (int16_t*)readBuffer;
    if (transfer_buffer_curr_len != 0) {
        Serial.printf("Error: Transfer buffer not empty. %d dropped\n", length);
        return;
    }
    // Drop every other sample (the second channel) while copying
    for(int i = 0; i < length; i++)
        transfer_buffer[i] = tempAudio[2 * i];
    transfer_buffer_curr_len = length;
}

void init_record(){
    // Sampling rate 16000Hz @ 16 bit resolution 
    // This is hardcoded in the code. Don't change.
    Audio.format(16000U, 16U);
}

void start_record(){
  Audio.startRecord(recordCallback);
}

void prediction_callback(float *vec, int len){
    int arg = argmax(vec, len);
    int oldarg = *(int*)q_oldest(&votingQ);
    if (oldarg >= NUM_LABELS || oldarg < 0)
        oldarg = 0;
    votingFrequence[arg]++;
    votingFrequence[oldarg]--;
    q_force_enqueue(&votingQ, &arg);
    if (votingFrequence[arg] >= VOTE_MAJORITY){
        char str[20];
        sprintf(str, "Pred: %s (%d)", labelInvArr[arg], arg);
        Screen.print(str, false);
    }
}


void setup(){
    q_init(&votingQ, votingContainer, VOTE_WIN_LEN, cb_write_int, cb_read_int);
    votingFrequence[0] = 5;
    Serial.begin(115200);
    Screen.init();
    delay(500);
    initFastRNN0();
    initFastRNN1();
    initFC();
    delay(500);
    Screen.clean();
    unsigned ret = sfastrnn2p_init(&fastrnnParams0,
        &fastrnnParams1, &fcParams, prediction_callback);
    Serial.printf("Return code: %d (init)\n", ret);
    if(ret != 0)
        error("Shallow FastRNN initialization failed (code %d)", ret);
    if(ret != 0) while(1);
    init_record();
    delay(500);
    Serial.println();
    Serial.println("Ready");
    Screen.print(0, "Ready");
}

int main(){
    // Setup the predictor thread and the
    // audio recording thread. The prediction
    // thread has alrady started and is waiting for audio.
    setup();
    delay(500);
    start_record();
    while (1){
        if (transfer_buffer_curr_len == 0){
            // For a 16, 16 fastRNN model, this can be pushed
            // 6ms without causing errors.
            rtos:wait_ms(5);
            continue;
        }
        unsigned ret = sfastrnn2p_add_new_samples(transfer_buffer,
            transfer_buffer_curr_len);
        if(ret != 0)
            Serial.printf("Error pushing to interface %d\n", ret);
        static int count = 0;
        count += transfer_buffer_curr_len;
        if(count % (128 * 1000) == 0)
            Serial.printf("Pushed %d seconds\n", (count/16000));
        transfer_buffer_curr_len = 0;
    }
}

