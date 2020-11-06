from enum import Enum, auto

import seedot.compiler.ast.ast as AST

import seedot.compiler.TF.Graph as Graph


# Contains code for each of the TF nodes encountered in the benchmarks.
# For each such TF node, outputs the corresponding SeeDot AST.

class TFNodesAST:
    class UninterpFuncCallNames(Enum):
        '''
        NOTE : SeeDot when compiling uninterpreted function calls, adds a new declaration for each uninterpreted function call.
        '''
        Input = auto()
        CreateCopy = auto()
        CreateIdentity = auto()
        CreateTensor = auto()
        # TODO : hack right now, for assign node :: fix this after discussing with Aseem
        CopyTensor = auto()
        Const = auto()
        Cast = auto()
        TruncatedNormal = auto()
        RandomUniform = auto()
        Tile = auto()
        MaxPool = auto()
        Pack = auto()
        Concat = auto()
        ExpandDims = auto()
        MaxPoolGrad = auto()
        Conv2DBackpropInput = auto()
        Conv2DBackpropFilter = auto()
        AvgPool = auto()
        Pad = auto()
        Squeeze = auto()
        TempFusedBatchNorm = auto()

    def getOperatorsIdx(token):
        # TODO : remove usage of this
        return AST.Operators.convSymbolToEnumValue(token)

    def MatMul(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        inp1Str = dictNodeNameToOutVarStr[inputsRef[0]]
        inp2Str = dictNodeNameToOutVarStr[inputsRef[1]]
        inp1AST = AST.ID(inp1Str)
        inp2AST = AST.ID(inp2Str)

        attrMapRef = curNode.getAttrMapRef()
        transposeABool = transposeBBool = False
        if ("\"transpose_a\"" in attrMapRef):
            transposeABool = attrMapRef["\"transpose_a\""].getB()
        if ("\"transpose_b\"" in attrMapRef):
            transposeBBool = attrMapRef["\"transpose_b\""].getB()
        if (transposeABool):
            inp1AST = AST.Transp(inp1AST)
        if (transposeBBool):
            inp2AST = AST.Transp(inp2AST)
        return (None, AST.BOp(inp1AST, TFNodesAST.getOperatorsIdx('*'), inp2AST))

    def Placeholder(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        #curNodeShapeLi = curNode.getAttrMapRef()["\"shape\""].getShape().getDimRef()
        curNodeShapeLi = extraNodeInfoDict[curNode.getName()][0]
        curNodeInputType = curNode.getAttrMapRef()["\"dtype\""].getDataType()
        assert(curNodeInputType is not Graph.DataTypeEnum.DT_INVALID)
        # TODO : There has to be some way to take range, understand the dimensions for SeeDot
        # CHANGESRI
        # return (None, AST.Input(curNodeShapeLi, curNodeInputType.name))
        return (None, AST.Decl(curNodeShapeLi, (-0.1, 0.1)))

    def Equal(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('=='),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
                              ))

    def Identity(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        # In SeeDot, J2=J1 creates a new reference for J1 -- so
        #	the corresponding code in Seedot cannot simply be J2 = J1.
        #	Instead create a new tensor first and then assign the old one to the new one.
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)

        curNodeDataType = curNode.getAttrMapRef()["\"T\""].getDataType()
        assert(curNodeDataType is not Graph.DataTypeEnum.DT_INVALID)

        curNodeShape = extraNodeInfoDict[curNode.getName()][0]
        retAST = AST.UninterpFuncCall(curNodeShape,
                                      TFNodesAST.UninterpFuncCallNames.CreateIdentity.name,
                                      [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])])
        return (None, retAST)

    def Add(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('+'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
                              ))

    def Mul(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('*'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
                              ))

    def Neg(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        return (None, AST.UOp(TFNodesAST.getOperatorsIdx('-'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])
                              ))

    def Sub(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('+'),
                              AST.UOp(TFNodesAST.getOperatorsIdx('-'),
                                      AST.ID(
                                  dictNodeNameToOutVarStr[inputsRef[1]])
        )))

    def Floor(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        return (None, AST.Func(TFNodesAST.getOperatorsIdx('floor'), AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])))

    def RealDiv(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('./'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
                              ))

    def FloorDiv(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        realDivAST = AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                             TFNodesAST.getOperatorsIdx('./'),
                             AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
                             )
        return (None, AST.Func(TFNodesAST.getOperatorsIdx('floor'), realDivAST))

    def VariableV2(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        curNodeShapeLi = curNode.getAttrMapRef(
        )["\"shape\""].getShape().getDimRef()[:]
        curNodeInputType = curNode.getAttrMapRef()["\"dtype\""].getDataType()

        # TODO_TAB : for inference, have commented out decl and inserted input nodes.
        # TODO : Right now in the current implementation, the dataType being passed to the node is being ignored by SeeDot.
        #		 Fix this later.
        # return (None, AST.Decl(curNodeShapeLi, curNodeInputType.name, None, None))
        # NOTE : since this becomes an input node right now, i have also added to be prefixed at top in ProcessTFGraph::prefixAllPlaceHolderNodes()
        # CHANGESRI
        # return (None, AST.Input(curNodeShapeLi, curNodeInputType.name))
        return (None, AST.Decl(curNodeShapeLi, [0.1, 0.1]))

    def Assign(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        curNodeShape = extraNodeInfoDict[curNode.getName()][0]

        # TODO_TAB : for inference, have commented the copyTensor function calls.
        # TODO : Hack -- fix this later after discussing with Aseem
        # return (None, AST.UninterpFuncCall(curNodeShape,
        # 									TFNodesAST.UninterpFuncCallNames.CopyTensor.name,
        # 									[AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
        # 									AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])]))

        return (None, None)

    def Const(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        assert(len(curNode.getInputsRef()) == 0)
        tensor = curNode.getAttrMapRef()["\"value\""].getTensor()
        curNodeDataType = curNode.getAttrMapRef()["\"dtype\""].getDataType()
        # create a different copy to not change the original copy
        curNodeShape = tensor.getShapeRef()[:]

        tensorConstantVal = tensor.getConstantVal()
        if tensorConstantVal is not None:
            # Use uinterpreted call of CreateTensor to create the tensor and fill it with a constant value
            dataPassed = None
            if curNodeDataType == Graph.DataTypeEnum.DT_INT32:
                dataPassed = AST.Int(tensorConstantVal, 32)
            elif curNodeDataType == Graph.DataTypeEnum.DT_FLOAT:
                dataPassed = AST.Float(tensorConstantVal)
            else:
                assert False

            if (len(curNodeShape) == 0):
                # This is a constant element
                retAST = dataPassed
            else:
                retAST = AST.UninterpFuncCall(curNodeShape,
                                              TFNodesAST.UninterpFuncCallNames.CreateTensor.name,
                                              [dataPassed],
                                              isSecret=False)
        else:
            # The tensor content is given as byte array. Extract val array from the byte array and create ast.
            if curNodeDataType == Graph.DataTypeEnum.DT_INT32:
                dataPassed = list(map(lambda x: AST.Int(
                    x, 32), tensor.getContentAsValArr()[:]))
            elif curNodeDataType == Graph.DataTypeEnum.DT_FLOAT:
                dataPassed = list(map(lambda x: AST.Float(
                    x), tensor.getContentAsValArr()[:]))
            else:
                assert False
            retAST = AST.Decl(curNodeShape, None, None,
                              dataPassed, isSecret=False)
        return (None, retAST)

    def Relu(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        return (None, AST.Func(TFNodesAST.getOperatorsIdx('relu'), AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])))

    def ApplyGradientDescent(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 3)
        inputTensor = AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])
        learningRate = AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
        deltaTensor = AST.ID(dictNodeNameToOutVarStr[inputsRef[2]])
        return (inputTensor,  AST.BOp(inputTensor,
                                      TFNodesAST.getOperatorsIdx('+'),
                                      AST.UOp(TFNodesAST.getOperatorsIdx('-'),
                                              AST.BOp(learningRate,
                                                      TFNodesAST.getOperatorsIdx(
                                                          '.*'),
                                                      deltaTensor))))

    def Shape(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        return (None, AST.Func(TFNodesAST.getOperatorsIdx('shape'), AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])))

    def Cast(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        sourceType = curNode.getAttrMapRef()["\"SrcT\""].getDataType()
        destType = curNode.getAttrMapRef()["\"DstT\""].getDataType()
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.Cast.name,
                                           [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                                            AST.ID(
                                               sourceType.name),
                                            AST.ID(
                                               destType.name)
                                            ]))

    def ZerosLike(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        curNodeOutputType = curNode.getAttrMapRef()["\"T\""].getDataType()
        assert(curNodeOutputType is not Graph.DataTypeEnum.DT_INVALID)
        retAST = AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                      TFNodesAST.UninterpFuncCallNames.CreateTensor.name,
                                      [AST.Int(0)],
                                      isSecret=False)
        return (None, retAST)

    def Fill(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        curNodeOutputShape = extraNodeInfoDict[inputsRef[0]][0]
        # inputsRef[0] denotes a shape and should have a rank of 1
        assert(len(curNodeOutputShape) == 1)

        curNodeOutputType = curNode.getAttrMapRef()["\"T\""].getDataType()
        assert(curNodeOutputType is not Graph.DataTypeEnum.DT_INVALID)

        retAST = AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                      TFNodesAST.UninterpFuncCallNames.CreateTensor.name,
                                      [AST.ID(
                                          dictNodeNameToOutVarStr[inputsRef[1]])],
                                      isSecret=False)
        return (None, retAST)

    def TruncatedNormal(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        curNodeDataType = curNode.getAttrMapRef()["\"dtype\""].getDataType()
        assert(curNodeDataType is not Graph.DataTypeEnum.DT_INVALID)
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        curNodeOutputShape = extraNodeInfoDict[curNode.getName()][0]
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.TruncatedNormal.name,
                                           [AST.ID(curNodeDataType.name)]
                                           + list(map(lambda x: AST.Int(x), curNodeOutputShape))
                                           ))  # TODO

    def RandomUniform(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        curNodeDataType = curNode.getAttrMapRef()["\"dtype\""].getDataType()
        assert(curNodeDataType is not Graph.DataTypeEnum.DT_INVALID)
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        curNodeOutputShape = extraNodeInfoDict[curNode.getName()][0]
        return (None, AST.UninterpFuncCall(curNodeOutputShape,
                                           TFNodesAST.UninterpFuncCallNames.RandomUniform.name,
                                           [AST.ID(curNodeDataType.name)]))

    def Maximum(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), TFNodesAST.getOperatorsIdx('max'), AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])))

    def Reshape(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.Reshape(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), extraNodeInfoDict[curNode.getName()][0], None))

    def Conv2D(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)

        options = {}
        # TODO : Parse other options and make sure backend is consuming those
        # Other options left to parse include T, data_format, dilations

        paddingUsed = curNode.getAttrMapRef()["\"padding\""].getS()
        if (paddingUsed == "\"SAME\""):
            options["padding"] = 0
        elif (paddingUsed == "\"VALID\""):
            options["padding"] = 1
        else:
            options["padding"] = -1

        stridesUsed = curNode.getAttrMapRef()["\"strides\""].getList().getILi()
        options["strides"] = stridesUsed

        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('#'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                              options))

    def MaxPool(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)

        options = {}

        stridesUsed = curNode.getAttrMapRef()["\"strides\""].getList().getILi()
        assert((stridesUsed[0] == 1) and (stridesUsed[3] == 1))
        strideH = stridesUsed[1]
        strideW = stridesUsed[2]

        kSizeUsed = curNode.getAttrMapRef()["\"ksize\""].getList().getILi()
        assert((kSizeUsed[0] == 1) and (kSizeUsed[3] == 1))
        kSizeH = kSizeUsed[1]
        kSizeW = kSizeUsed[2]

        paddingUsedStr = curNode.getAttrMapRef()["\"padding\""].getS()
        zPadH = zPadW = -1
        if (paddingUsedStr == "\"SAME\""):
            zPadH = int((kSizeH - 1)/2)
            zPadW = int((kSizeW - 1)/2)
        elif (paddingUsedStr == "\"VALID\""):
            zPadH = zPadW = 0
        else:
            zPadH = zPadW = -1

        inputShape = extraNodeInfoDict[inputsRef[0]][0]
        imgH = inputShape[1]
        imgW = inputShape[2]
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.MaxPool.name,
                                           [AST.Int(kSizeH, 32), AST.Int(kSizeW, 32),
                                            AST.Int(zPadH, 32), AST.Int(
                                                zPadW, 32),
                                            AST.Int(strideH, 32), AST.Int(
                                               strideW, 32),
                                            AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])]
                                           ))

    def Pack(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        N = curNode.getAttrMapRef()["\"N\""].getI()
        axis = curNode.getAttrMapRef()["\"axis\""].getI()
        assert(len(inputsRef) == N)
        retAST = AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                      TFNodesAST.UninterpFuncCallNames.Pack.name,
                                      list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef)) + [AST.Int(axis)])
        return (None, retAST)

    def ConcatV2(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        N = curNode.getAttrMapRef()["\"N\""].getI()
        assert(len(inputsRef) == N+1)  # One extra for axis
        # TODO : Since the axis of concat is constant, therefore, its known here - the input's sizes along that dim should be
        #		passed as input to the below function.
        #		For now hardcoding.
        retAST = AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                      TFNodesAST.UninterpFuncCallNames.Concat.name +
                                      str(N) + 'T',
                                      list(map(lambda x: AST.ID(
                                          dictNodeNameToOutVarStr[x]), inputsRef)),
                                      outputDiffInpDims=1
                                      )
        return (None, retAST)

    def ExpandDims(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        retAST = AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                      TFNodesAST.UninterpFuncCallNames.ExpandDims.name,
                                      list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef)))
        return (None, retAST)

    def Slice(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 3)
        curNodeDataType = curNode.getAttrMapRef()["\"T\""].getDataType()
        curNodeShapeASTLi = list(map(lambda x: AST.Int(
            x), extraNodeInfoDict[curNode.getName()][0]))
        retAST = AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                      TFNodesAST.UninterpFuncCallNames.CreateCopy.name,
                                      [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),  # of this
                                       # begin idx
                                       AST.ID(
                                           dictNodeNameToOutVarStr[inputsRef[1]]),
                                       # size
                                       AST.ID(
                                           dictNodeNameToOutVarStr[inputsRef[2]])
                                       ])
        return (None, retAST)

    def Tile(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.Tile.name,
                                           [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]), AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])]))

    def ShapeN(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        # TODO : generalize -- remove usage of Declare
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        N = curNode.getAttrMapRef()["\"N\""].getI()
        assert(N == 2)
        # TODO
        # curNodeShape = extraNodeInfoDict[curNode.getName()][0]
        # curNodeDataType = curNode.getAttrMapRef()["\"T\""].getDataType()
        # retAST = AST.Let(AST.ID('temp_shapen_1'), AST.Declare(list(map(lambda x : AST.Int(x), curNodeShape)), AST.ID(curNodeDataType.name)), None)
        # retAST.expr = AST.Let(AST.Index(AST.ID('temp_shapen_1'), [AST.Int(0)]),
        # 					  AST.Func(TFNodesAST.getOperatorsIdx('shape'), AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])),
        # 					  None)
        # retAST.expr.expr = AST.Let(AST.Index(AST.ID('temp_shapen_1'), [AST.Int(1)]),
        # 						   AST.Func(TFNodesAST.getOperatorsIdx('shape'), AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])),
        # 						   AST.ID('temp_shapen_1'))

        return (None, None)

    def Sum(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.Reduce(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                                 AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                                 TFNodesAST.getOperatorsIdx('+')))

    def Prod(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.Reduce(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                                 AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                                 TFNodesAST.getOperatorsIdx('*')))

    def Mean(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.Reduce(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                                 AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                                 TFNodesAST.getOperatorsIdx('mean')))

    def ArgMax(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.ArgMax(extraNodeInfoDict[curNode.getName()][0],
                                 AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                                 AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                                 extraNodeInfoDict[inputsRef[0]][0]))

    def LogSoftmax(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        expAST = AST.Func(TFNodesAST.getOperatorsIdx('exp'),
                          AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]))
        reduceAST = AST.Reduce(expAST, AST.Int(-1),
                               TFNodesAST.getOperatorsIdx('+'))
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('+'),
                              AST.UOp(TFNodesAST.getOperatorsIdx('-'), AST.Func(TFNodesAST.getOperatorsIdx('log'), reduceAST))))

    def StopGradient(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        return (None, AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]))

    def SoftmaxCrossEntropyWithLogits(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        # Input1 is logits and Input2 is the one-hot encoding true distribution
        # Calculate softmax on input1 and cross-entropy between that (p(x)) and true-distribution (q(x))
        # Cross-entropy = \summation_x{-q(x)*log(p(x))}
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        logitsInpt = AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])
        labelsInpt = AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])

        # reduce along column to get row-vector
        # TODO : softmax or implement here ?
        retAST = AST.Let(AST.ID('temp_softmax'), AST.Func(
            TFNodesAST.getOperatorsIdx('softmax'), logitsInpt), None)
        retAST.expr = AST.Let(AST.ID('temp_1'),
                              AST.UOp(TFNodesAST.getOperatorsIdx('-'),
                                      AST.Reduce(AST.BOp(labelsInpt,
                                                         TFNodesAST.getOperatorsIdx(
                                                             '.*'),
                                                         AST.Func(TFNodesAST.getOperatorsIdx('log'), AST.ID('temp_softmax'))),
                                                 1, TFNodesAST.getOperatorsIdx('+'))),
                              AST.ID('temp_1'))
        return (None, retAST)

    def BroadcastGradientArgs(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        return (None, AST.ID("temp"))  # TODO

    def ReluGrad(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.Cond(AST.ID(dictNodeNameToOutVarStr[inputsRef[1]]),
                               AST.Int(1),
                               AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                               AST.Int(0)))

    def MaxPoolGrad(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.MaxPoolGrad.name,
                                           list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef))))

    def Conv2DBackpropInput(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.Conv2DBackpropInput.name,
                                           list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef))))

    def Conv2DBackpropFilter(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.Conv2DBackpropFilter.name,
                                           list(map(lambda x: AST.ID(dictNodeNameToOutVarStr[x]), inputsRef))))

    def NoOp(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        return (None, None)

    def Square(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('.*'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])
                              ))

    def AvgPool(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 1)

        options = {}

        stridesUsed = curNode.getAttrMapRef()["\"strides\""].getList().getILi()
        assert((stridesUsed[0] == 1) and (stridesUsed[3] == 1))
        strideH = stridesUsed[1]
        strideW = stridesUsed[2]

        kSizeUsed = curNode.getAttrMapRef()["\"ksize\""].getList().getILi()
        assert((kSizeUsed[0] == 1) and (kSizeUsed[3] == 1))
        kSizeH = kSizeUsed[1]
        kSizeW = kSizeUsed[2]

        paddingUsedStr = curNode.getAttrMapRef()["\"padding\""].getS()
        zPadH = zPadW = -1
        if (paddingUsedStr == "\"SAME\""):
            zPadH = int((kSizeH - 1)/2)
            zPadW = int((kSizeW - 1)/2)
        elif (paddingUsedStr == "\"VALID\""):
            zPadH = zPadW = 0
        else:
            zPadH = zPadW = -1

        inputShape = extraNodeInfoDict[inputsRef[0]][0]
        imgH = inputShape[1]
        imgW = inputShape[2]
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.AvgPool.name,
                                           [AST.Int(kSizeH, 32), AST.Int(kSizeW, 32),
                                            AST.Int(zPadH, 32), AST.Int(
                                                zPadW, 32),
                                            AST.Int(strideH, 32), AST.Int(
                                               strideW, 32),
                                            AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])]
                                           ))

    def Pad(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        # Mode refers to 'CONSTANT', 'REFLECT' or 'SYMMETRIC'
        mode = 0
        if ("\"mode\"" in curNode.getAttrMapRef()):
            mode = curNode.getAttrMapRef()["\"mode\""].getI()

        constant_values = 0
        if ("\"constant_values\"" in curNode.getAttrMapRef()):
            constant_values = curNode.getAttrMapRef()[
                "\"constant_values\""].getI()

        # For now to make life easy - deal with SYMMETRIC AND REFLECT when time comes
        assert(mode == 0 and constant_values == 0)
        inputsRef = curNode.getInputsRef()
        inputTensorShapeLi = extraNodeInfoDict[inputsRef[0]][0]
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.Pad.name,
                                           [
            AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
            AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
        ],
            outputDiffInpDims=1
        ))

    def FusedBatchNorm(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        # NOTE : Since the weights to this layer will be scaled appropriately, this op will become identity.
        inputsRef = curNode.getInputsRef()

        # TODO : This below thing is the right way of implementing the operator
        #		For now using uninterpreted function call.
        # tempAst = AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
        # 					TFNodesAST.getOperatorsIdx('*'),
        # 					AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
        # 					)
        # return (None, AST.BOp(tempAst,
        # 					TFNodesAST.getOperatorsIdx('+'),
        # 					AST.ID(dictNodeNameToOutVarStr[inputsRef[2]])
        # 					))
        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.TempFusedBatchNorm.name,
                                           [AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                                            AST.ID(
                                               dictNodeNameToOutVarStr[inputsRef[1]]),
                                            AST.ID(
                                               dictNodeNameToOutVarStr[inputsRef[2]]),
                                            ]
                                           ))

    def Squeeze(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        # TODO : Do this in somewhat better way
        inputsRef = curNode.getInputsRef()
        inputTensorShape = extraNodeInfoDict[inputsRef[0]][0]
        inputTensorRank = len(inputTensorShape)

        squeezeDims = curNode.getAttrMapRef(
        )["\"squeeze_dims\""].getList().getILi()
        squeezeDimsRank = len(squeezeDims)

        return (None, AST.UninterpFuncCall(extraNodeInfoDict[curNode.getName()][0],
                                           TFNodesAST.UninterpFuncCallNames.Squeeze.name,
                                           list(map(lambda x: AST.Int(x, 32), squeezeDims)) +
                                           [
            AST.ID(dictNodeNameToOutVarStr[inputsRef[0]])
        ]
        ))

    def BiasAdd(graph: Graph.Graph, curNode: Graph.Node, dictNodeNameToOutVarStr: dict, extraNodeInfoDict: dict):
        inputsRef = curNode.getInputsRef()
        assert(len(inputsRef) == 2)
        return (None, AST.BOp(AST.ID(dictNodeNameToOutVarStr[inputsRef[0]]),
                              TFNodesAST.getOperatorsIdx('+'),
                              AST.ID(dictNodeNameToOutVarStr[inputsRef[1]])
                              ))
