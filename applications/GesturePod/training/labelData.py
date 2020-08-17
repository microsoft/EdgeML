'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.

labelData.py used to label (sliding) window of data.
'''

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

module_path = os.path.abspath(os.path.join('../'))
if module_path not in sys.path:
    sys.path.append(module_path)
from timestep.plotterobjects import LinePlotter, BarPlotter
from timestep.plotterobjects import BooleanPlotter, StatusBox
from timestep.eventhandler import BasicEventHandler

SILENCE = '0'
NOISE = '1'
DTAP = '3'
RIGHT_TWIST = '4'
LEFT_TWIST = '5'
TWIRL = '7'
DOUBLE_SWIPE = '9'

def defaultCB(event, eventHandlerObj):
    eobj = eventHandlerObj
    if eobj.CURRENT_STATE is not 'PAUSED':
        eobj.startIndex += eobj.stride
        eobj.endIndex += eobj.stride
    if eobj.startIndex < 0:
        eobj.startIndex = 0
        eobj.endIndex = eobj.length
    dataFrame = eventHandlerObj.dataFrame
    if eobj.endIndex >= len(dataFrame):
        eobj.endIndex = len(dataFrame)
        eobj.startIndex = eobj.endIndex - eobj.length
        eventHandlerObj.statusMessage += "End of file\n"


def leftRightCB(event, eventHandlerObj):
    eobj = eventHandlerObj
    if event == 'left':
        eobj.startIndex -= eobj.stride
        eobj.endIndex -= eobj.stride
    elif event == 'right':
        eobj.startIndex += eobj.stride
        eobj.endIndex += eobj.stride

    dataFrame = eventHandlerObj.dataFrame
    if eobj.startIndex < 0:
        eobj.startIndex = 0
        eobj.endIndex = eobj.length
    if eobj.endIndex > len(dataFrame):
        eobj.endIndex = len(dataFrame)
        eobj.startIndex = eobj.endIndex - eobj.length


def quitCB(event, eventHandlerObj):
    global fileName
    if not os.path.exists('data/labelled_data'):
        os.mkdir('data/labelled_data')
    outfilename = "./data/labelled_data/" + fileName[:-4] + '_labelled.csv'
    print("Saving and exiting")
    print("Outputfile: %s" % outfilename)
    eventHandlerObj.dataFrame.to_csv(outfilename, index=False)
    exit()


def labelCB(event, eventHandlerObj):
    eobj = eventHandlerObj
    if eobj.CURRENT_STATE is not 'PAUSED':
        eobj.statusMessage += "Labelling allowed only when paused.\n"
        return
    df = eobj.dataFrame
    s = eobj.startIndex
    e = eobj.endIndex
    labelCol = eobj.labelCol
    labelCol = df.columns.get_loc(labelCol)
    eobj.dataFrame.iloc[s:e, labelCol] = int(event)


def spacePressCB(event, eventHandlerObj):
    CURRENT_STATE = eventHandlerObj.CURRENT_STATE
    if CURRENT_STATE == 'PAUSED':
        eventHandlerObj.CURRENT_STATE = 'PLAY'
    else:
        eventHandlerObj.CURRENT_STATE = 'PAUSED'


def run(dataFrame, length, stride):
    fig = plt.figure()
    #AX
    accXPlotter = LinePlotter(length=length, subplotIndex=331,
                              ylim=[-20000, 5000])
    accXPlotter.ax.set_title("acc_x")
    #AY
    accYPlotter = LinePlotter(length=length, subplotIndex=332,
                              ylim=[-20000, 5000])
    accYPlotter.ax.set_title("acc_y")
    #AZ
    accZPlotter = LinePlotter(length=length, subplotIndex=333,
                              ylim=[-20000, 5000])
    accZPlotter.ax.set_title("acc_z")
    #GX
    gyrXPlotter = LinePlotter(length=length, subplotIndex=334,
                              ylim=[-600, 600])
    gyrXPlotter.ax.set_title("gyr_x")
    #GY
    gyrYPlotter = LinePlotter(length=length, subplotIndex=335,
                              ylim=[-600, 600])
    gyrYPlotter.ax.set_title("gyr_y")
    #GZ
    gyrZPlotter = LinePlotter(length=length, subplotIndex=336,
                              ylim=[-600, 600])
    gyrZPlotter.ax.set_title("gyr_z")
    #Manual Label
    manualLabelPlotter = LinePlotter(length=length, subplotIndex=337,
                                     ylim=[-0.2, 9.2],
                                     c='r')
    manualLabelPlotter.ax.set_title("label")
    #Toggle
    togglePlotter = LinePlotter(length=length, subplotIndex=338,
                                ylim=[-0.2, 9.2],
                                c='r')
    togglePlotter.ax.set_title("toggle")
    statusBox = StatusBox(fig, initText='PLAY')
    plotterList = [accYPlotter, gyrYPlotter, manualLabelPlotter]
    plotterList += [togglePlotter, statusBox]
    eventHandler = BasicEventHandler(plotterList, fig)
    eventHandler.CURRENT_STATE = 'PLAY'
    eventHandler.startIndex = 0
    eventHandler.endIndex = length
    eventHandler.length = length
    eventHandler.stride = stride
    eventHandler.statusMessage = ""
    eventHandler.maxLength = len(dataFrame)
    eventHandler.registerEvent(' ', spacePressCB)
    eventHandler.deregisterEvent('close_event')
    eventHandler.registerEvent('close_event', quitCB)
    eventHandler.registerEvent('q', quitCB)
    eventHandler.registerEvent('default', defaultCB)
    eventHandler.registerEvent('right', leftRightCB)
    eventHandler.registerEvent('left', leftRightCB)
    # Labelling events
    eventHandler.registerEvent(DTAP, labelCB)
    eventHandler.registerEvent(LEFT_TWIST, labelCB)
    eventHandler.registerEvent(RIGHT_TWIST, labelCB)
    eventHandler.registerEvent(NOISE, labelCB)
    eventHandler.registerEvent(SILENCE, labelCB)
    eventHandler.registerEvent(TWIRL, labelCB)
    eventHandler.registerEvent(DOUBLE_SWIPE, labelCB)
    labelCol = 'mlabel'
    if labelCol not in dataFrame:
        dataFrame[labelCol] = 0
    eventHandler.dataFrame = dataFrame
    eventHandler.labelCol = labelCol
    labelColIndex = dataFrame.columns.get_loc(labelCol)
    axColIndex = dataFrame.columns.get_loc('ax')
    ayColIndex = dataFrame.columns.get_loc('ay')
    azColIndex = dataFrame.columns.get_loc('az')
    gxColIndex = dataFrame.columns.get_loc('gx')
    gyColIndex = dataFrame.columns.get_loc('gy')
    gzColIndex = dataFrame.columns.get_loc('gz')
    toggleIndex = dataFrame.columns.get_loc('toggle')
    while True:
        startIndex = eventHandler.startIndex
        endIndex = eventHandler.endIndex
        CURRENT_STATE = eventHandler.CURRENT_STATE
        # AX
        sensorValues = dataFrame.iloc[startIndex:endIndex, axColIndex]
        accXPlotter.setValues(sensorValues)
        # AY
        sensorValues = dataFrame.iloc[startIndex:endIndex, ayColIndex]
        accYPlotter.setValues(sensorValues)
        # Az
        sensorValues = dataFrame.iloc[startIndex:endIndex, azColIndex]
        accZPlotter.setValues(sensorValues)
        # GX
        sensorValues = dataFrame.iloc[startIndex:endIndex, gxColIndex]
        gyrXPlotter.setValues(sensorValues)
        # GY
        sensorValues = dataFrame.iloc[startIndex:endIndex, gyColIndex]
        gyrYPlotter.setValues(sensorValues)
        # GZ
        sensorValues = dataFrame.iloc[startIndex:endIndex, gzColIndex]
        gyrZPlotter.setValues(sensorValues)
        # Labelling
        label = dataFrame.iloc[startIndex:endIndex, labelColIndex]
        manualLabelPlotter.setValues(label)
        # Toggle info
        toggleVals = dataFrame.iloc[startIndex:endIndex, toggleIndex]
        togglePlotter.setValues(toggleVals)
        # Update graph
        eventHandler.handleQueuedEvents()
        statusMessage = eventHandler.statusMessage
        statusMessage += CURRENT_STATE
        statusBox.setText(statusMessage)
        eventHandler.updatePlot()
        eventHandler.statusMessage = ""
    return dataFrame

fileName = fileName  = sys.argv[1]
def main():
    global fileName
    file      = "./data/raw_data/" + fileName 
    dataFrame = pd.read_csv(file)
    run(dataFrame, 400, 20)


if __name__=="__main__":
    main()
