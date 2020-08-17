'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
 
csvFromSerial.py reads data from Serial COM port and stores it as a csv. 
This script is used for collecting raw data.
'''

import serial
import sys
from threading import Thread
import os

TOGGLESTATUS = 'OFF'
EXIT = False

'''
Toggle can be used to mark the approximate beginng and end boundries for labels.
'''
def threadToggle():
    global TOGGLESTATUS
    global EXIT
    while not EXIT:
        try:
            line = input()
            if ' ' in line:
                if TOGGLESTATUS == 'OFF':
                    TOGGLESTATUS = 'ON'
                else:
                    TOGGLESTATUS = 'OFF'
            if 'q' in line:
                EXIT = True
                break
        except KeyboardInterrupt:
            print("\nExiting.")


def main():
    global EXIT
    global TOGGLESTATUS
    sessionKey = ''
    outFile = ''
    comPort = 'COM13'
    baudRate = 115200
    if len(sys.argv) < 2:
        print("Usage: %s SESSION_KEY [COM Port]" % (sys.argv[0]))
        EXIT = True
        return
    else:
        sessionKey = sys.argv[1]
        if len(sys.argv) > 2:
            comPort = sys.argv[2]
        outFile = sessionKey + '.csv'

    ser = serial.Serial(port=comPort, baudrate=baudRate, timeout=1)
    testChar = ser.read()
    if len(testChar) < 1:
        print("No bytes received. Exiting!")
        return
    # Ignore what ever remains of the first line
    for x in range(0, 5):
        a = ser.readline()
    # Create directory
    if not os.path.exists('data'):
        os.mkdir('data')
    if not os.path.exists('data/raw_data'):
        os.mkdir('data/raw_data')
    outFile = './data/raw_data/' + outFile
    fout = open(outFile, 'w')
    fout.write("millis,ax,ay,az,gx,gy,gz,toggle\n")
    print("Starting fetch")
    print("Use spacebar to toggle label On/Off")
    linesWritten = 0
    while not EXIT:
        try:
            print('\r%-3s' % (TOGGLESTATUS), end='')
            a = ser.readline().decode('utf-8')
            a = a[:-3]
            tokens = a.split(',')
            if len(tokens) != 7:
                continue
            if TOGGLESTATUS == 'OFF':
                toggle = 0
            else:
                toggle = 1
            a += ',' + str(toggle)
            a += '\n'
            fout.write(a)
            linesWritten += 1
            if linesWritten % 500 is 0:
                print("\r%-3s%5d" % (TOGGLESTATUS, linesWritten), end='')
        except KeyboardInterrupt:
            ser.close()
            fout.close()
            print("\nExiting.")
            EXIT = True
            break


if __name__ == '__main__':
    thread1 = Thread(target = threadToggle)
    thread2 = Thread(target = main)
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()
