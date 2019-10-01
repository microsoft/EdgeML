'''
Copyright (c) Microsoft Corporation. All rights reserved.
Licensed under the MIT license.
'''

import matplotlib.pyplot as plt
import numpy as np


class LinePlotter():

    def __init__(self, length, subplotIndex=None,
                 ax=None, initValues = None,
                 ylim=None, **kwargs):
        self.length = length
        if all(v is None for v in {subplotIndex, ax}):
            raise ValueError("Expected either ax or subplotIndex")
        if ax is None:
            self.subplotIndex = subplotIndex
            self.ax = plt.subplot(subplotIndex)
        else:
            self.ax = ax
        if ylim is not None:
            self.ax.set_ylim(ylim)
        if initValues is None:
            initValues = np.random.rand(length)
        self.setValues(initValues)
        self.line, = self.ax.plot(self.values, **kwargs)

    def updatePlot(self):
        self.line.set_ydata(self.values)

    def setValues(self, values):
        assert(len(values) == self.length)
        self.values = values


class BarPlotter():
    def __init__(self, numBars, subplotIndex=None,
                 ax = None, ticks=None,
                 initValues=None, ylim=None, **kwargs):
        self.length = numBars
        if all(v is None for v in {subplotIndex, ax}):
            raise ValueError("Expected either ax or subplotIndex")
        if ax is None:
            self.subplotIndex = subplotIndex
            self.ax = plt.subplot(subplotIndex)
        else:
            self.ax = ax
        if ticks is None:
            ticks = np.arange(numBars)
        if ylim is not None:
            self.ax.set_ylim(ylim)
        if initValues is None:
            initValues = np.random.rand(numBars)
        self.setValues(initValues)
        self.line = self.ax.bar(ticks, self.values, **kwargs)

    def updatePlot(self):
        for i in range(len(self.line.patches)):
            self.ax.patches[i].set_height(self.values[i])

    def setValues(self, values):
        assert(len(values) == self.length)
        self.values = values


class BooleanPlotter():
    def __init__(self, length, subplotIndex=None,
                 ax = None, initValues = None,
                 ylim=None, **kwargs):
        self.length = length
        if all(v is None for v in {subplotIndex, ax}):
            raise ValueError("Expected either ax or subplotIndex")
        if ax is None:
            self.subplotIndex = subplotIndex
            self.ax = plt.subplot(subplotIndex)
        else:
            self.ax = ax
        if ylim is not None:
            self.ax.set_ylim(ylim)
        else:
            self.ax.set_ylim([-0.1, 1.1])
        if initValues is None:
            initValues = np.random.rand(length)
        self.setValues(initValues)
        if not ('linestyle' in kwargs or 'ls' in kwargs):
            kwargs['linestyle'] = 'None'
        if not ('markersize' in kwargs or 'ms' in kwargs):
            kwargs['markersize'] = 10.00
        if not ('marker' in kwargs):
            kwargs['marker'] = 'o'
        if not ('color' in kwargs or 'c' in kwargs):
            kwargs['color'] = 'r'
        self.line, = self.ax.plot(self.values, **kwargs)

    def updatePlot(self):
        self.line.set_ydata(self.values)

    def setValues(self, values):
        assert(len(values) == self.length)
        self.values = values


class StatusBox:
    def __init__(self, fig, x=None, y=None, initText=None):
        if x is None:
            x = 0.01
            y = 0.01
        self.x = x
        self.y = y
        if initText is None:
            initText = 'Status:'
        self.text = initText
        self.textbox = fig.text(self.x, self.y, initText)

    def setText(self, s):
        self.text = s

    def updatePlot(self):
        self.textbox.set_text(self.text)
