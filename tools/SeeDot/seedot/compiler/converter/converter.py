# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

import os

from seedot.compiler.converter.bonsai import *
from seedot.compiler.converter.protonn import *
from seedot.compiler.converter.util import *

import seedot.common as Common

# Main file which sets the configurations and creates the corresponding object


class Converter:

    def __init__(self, algo, version, datasetType, target, datasetOutputDir, outputDir):
        setAlgo(algo)
        setVersion(version)
        setDatasetType(datasetType)
        setTarget(target)

        # Set output directories
        setDatasetOutputDir(datasetOutputDir)
        setOutputDir(outputDir)

    def setInput(self, modelDir, trainingInput, testingInput):
        setModelDir(modelDir)

        # Type of normalization: 0 - No norm, 1 - MinMax norm, 2 - L2 norm, 3 -
        # MeanVar norm
        if os.path.isfile(os.path.join(modelDir, "minMaxParams")):
            setNormType(1)
        else:
            setNormType(0)

        setDatasetInput(trainingInput, testingInput)

        self.inputSet = True

    def run(self):
        if self.inputSet != True:
            raise Exception("Set input paths before running Converter")

        algo, version = getAlgo(), getVersion()

        if algo == Common.Algo.Bonsai and version == Common.Version.Fixed:
            obj = BonsaiFixed()
        elif algo == Common.Algo.Bonsai and version == Common.Version.Float:
            obj = BonsaiFloat()
        elif algo == Common.Algo.Protonn and version == Common.Version.Fixed:
            obj = ProtonnFixed()
        elif algo == Common.Algo.Protonn and version == Common.Version.Float:
            obj = ProtonnFloat()
        else:
            assert False

        obj.run()
