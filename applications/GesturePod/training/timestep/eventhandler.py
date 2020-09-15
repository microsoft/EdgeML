'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
'''

import matplotlib.pyplot as plt
from timestep.plotterobjects import LinePlotter
import numpy as np


class BasicEventHandler:
    '''
    Multiple instances of the event handler can't
    be used and is not intended to be used as such.
    '''

    def __init__(self, plotterList, fig, delay=0.00000001):
        self.plotterList = plotterList
        self.eventList = []
        self.fig = fig
        self.fig.canvas.mpl_connect('key_press_event', self.keyPress)
        self.fig.canvas.mpl_connect('close_event', self.closeEvent)
        self.eventCallbackDict = {
            'close_event': self.closeEventCB,
        }
        self.exitOnCloseEvent = True
        self.delay = delay

    def registerEvent(self, key, callback):
        if key in self.eventCallbackDict:
            raise ValueError("An event already registerd for %s" % key)
        self.eventCallbackDict[key] = callback

    def deregisterEvent(self, key):
        if key not in self.eventCallbackDict:
            return
        del self.eventCallbackDict[key]

    def updatePlot(self):
        plotterList = self.plotterList
        for plotter in plotterList:
            plotter.updatePlot()
        plt.draw()
        plt.pause(self.delay)

    def setExitOnCloseEvent(self, val):
        if val:
            self.exitOnCloseEvent = True
        else:
            self.exitOnCloseEvent = False
        return self.exitOnCloseEvent

    def handleQueuedEvents(self):
        while len(self.eventList) > 0:
            event = self.eventList[0]
            del self.eventList[0]
            if event not in self.eventCallbackDict:
                continue
            handler = self.eventCallbackDict[event]
            handler(event, self)
        if 'default' in self.eventCallbackDict:
            handler = self.eventCallbackDict['default']
            handler('default', self)

    def keyPress(self, event):
        self.eventList.append(event.key)

    def closeEvent(self, event):
        self.eventList.append(event.name)

    def closeEventCB(self, event, obj):
        if self.exitOnCloseEvent:
            exit()
        self.fig.close()


def test():
    fig = plt.figure()
    l2 = LinePlotter(10, 111)
    eventHandler = BasicEventHandler([l2], fig, currStat='1')
    eventHandler.setExitOnCloseEvent(True)
    while True:
        values = np.random.rand(10)
        l2.setValues(values)
        eventHandler.updatePlot()

