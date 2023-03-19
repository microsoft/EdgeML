/*
 * Step 1: Record data for 2 seconds
 * Step 2: Send data through serial
 * Step 3: Play data in python and see if its okay
 * 
  // Only 16bit is supported by the audio library. Comment here[1]
  // The underlying BSP audio calls to mbed-os used here[2], defined here[3]
  // and documented here[4] seems to support 16000Hz @16 bits
  // [1] https://github.com/Microsoft/devkit-sdk/blob/d8f9c2cf1a26aa44b8aae13e3ef7247f3f730711/AZ3166/src/libraries/AudioV2/src/AudioClassV2.cpp#47
  // [2] https://github.com/Microsoft/devkit-sdk/blob/d8f9c2cf1a26aa44b8aae13e3ef7247f3f730711/AZ3166/src/libraries/AudioV2/src/AudioClassV2.cpp#L101
  // [3] https://github.com/Microsoft/devkit-sdk/blob/d8f9c2cf1a26aa44b8aae13e3ef7247f3f730711/AZ3166/src/libraries/AudioV2/src/stm32412g_discovery_audio.h#L295
  // [4] https://os.mbed.com/users/the_sz/code/BSP_DISCO_F746NG_patch_fixed/docs/a4e658110084/group__STM32746G__DISCOVERY__AUDIO__Out__Private__Functions.html#ga18576073e3e3aca86934fc98288bc83a 
 */

#include <arm_math.h>
#include <Arduino.h>
#include <OledDisplay.h>
#include <RingBuffer.h>
#include <AudioClassV2.h>
#include <circularq.h>

// Must be a multiple of 512 hence 16384 instead of 16000.
// The 512 requirement is because of the block size used
// to push to the priority queue. Partial pushes fail.
// Roughly 1 second audio
#define AUDIO_SIZE (16384 * 3)
// 2 times as we are using 16bit (2 char vs 1 char)
#define AUDIO_BUFFER_SIZE (2 * AUDIO_SIZE)

static AudioClass& Audio = AudioClass::getInstance();
static char audio_container[AUDIO_BUFFER_SIZE];
static FIFOCircularQ audioQ;
char readBuffer[AUDIO_CHUNK_SIZE];


void recordCallback(void) {
  int length = Audio.readFromRecordBuffer(readBuffer, AUDIO_CHUNK_SIZE);
  q_enqueue_batch(&audioQ, (void *)readBuffer, sizeof(char), length);
}

void printIdleMessage(){
  Screen.clean();
  Screen.print(0, "Audio Test");
  Screen.print(1, "Hold A to Record", true);
}

void record(){
  Serial.println("Start recording");
  // Sampling rate 16000Hz @ 16 bit resolution 
  Audio.format(16000U, 16U);
  Audio.startRecord(recordCallback);
}

void setup(void){
  pinMode(LED_BUILTIN, OUTPUT);
  Serial.begin(115200);
  Screen.init();
  Serial.println("Testing 16bit@16000Hz audio");
  q_init(&audioQ, (void *)audio_container, AUDIO_BUFFER_SIZE,
        cb_write_char, cb_read_char);
  // Initialize the button pin as a input
  pinMode(USER_BUTTON_A, INPUT);
  printIdleMessage();
  delay(500);
}

void loop(void){
  printIdleMessage();
  while(digitalRead(USER_BUTTON_A));
    
  Screen.clean();
  Screen.print(0, "Start recording:");
  record();
  while(!(q_is_full(&audioQ))){
    delay(30);
  }
  if(Audio.getAudioState() == AUDIO_STATE_RECORDING){
    Audio.stop();
  }
  Screen.print(0, "Recording done.");
  Screen.print(1, "Printing to Serial.", true);
  for(int i = 0; i < AUDIO_BUFFER_SIZE; i=i+2){
    // Note that the audio returned is fake sterio, that is
    // the same channel is repeated and interleaved to create
    // a sterio effect. Hence we skip every other sample.
    int16_t t = *(int16_t *)q_atN(&audioQ, i);
    if (i % 4 == 0)
      Serial.printf("%d, ", t);
    if(i % 40 == 0){
      Serial.println();
      delay(10);
    }
  }
  delay(100);
  Serial.println("Done");
  Screen.clean();
  q_reset(&audioQ);
  delay(100);
}

int main(){
  setup();
  Serial.println("Starting");
  for(int i = 0; i < 100; i++)
    loop();
    delay(500);
}