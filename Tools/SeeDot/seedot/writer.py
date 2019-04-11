# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.


class Writer:

    def __init__(self, fileName):
        self.file = open(fileName, 'w')
        self.indentLevel = 0

    def printf(self, str, *args, indent=False):
        if indent:
            self.file.write('\t' * self.indentLevel)
        self.file.write(str % args)

    def increaseIndent(self):
        self.indentLevel += 1

    def decreaseIndent(self):
        self.indentLevel -= 1

    def close(self):
        self.file.close()
