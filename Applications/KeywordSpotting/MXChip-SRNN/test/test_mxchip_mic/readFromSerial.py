from serial import Serial
import scipy.io.wavfile as wav
import numpy as np
 
port = "COM3"
baud = 115200
samplingrate = 16000

ser = Serial(port, baud, timeout=1)
if ser.isOpen():
     print(ser.name + ' is open...')

lineList = []
i=0
while True:
    print("\r Lines read: %5d" % len(lineList), end='')
    lin = ser.readline()
    lin = lin.decode("utf-8")
    lin = lin.strip()
    if len(lin) == 0:
        continue
    if "Done" in lin:
        break
    lineList.append(lin)

ser.close()
print()

# Skip the first two and last two lines for good measure
longStr = ''.join(lineList[2:-2])
longStr = longStr.replace(' ', '')
numList = np.array(longStr.split(',')[:-1]).astype(float)
print("Writing %d (%fs) values" % (len(numList), len(numList)/samplingrate))
wav.write('output.wav', samplingrate, numList)