
import os
import pickle
import sys

import seedot.compiler.ast.ast as AST
from seedot.compiler.ast.printAST import PrintAST
from seedot.compiler.ast.mtdAST import MtdAST

import seedot.compiler.TF.Graph as Graph
from seedot.compiler.TF.TFNodesAST import TFNodesAST


def checkTFNodeNameForEq(curNodeOp: str, givenOp: str):
    return (curNodeOp == "\"" + givenOp + "\"")


def generateASTForNode(graph, curNode, dictNodeNameToOutVarStr, extraNodeInfoDict):
    # print("===>>> Generating AST for (nodeOp, nodeName) : (" + curNode.getOp() + ", " + curNode.getName() + ")")
    curNodeOp = curNode.getOp()
    ast = None
    # To remove the " at the begin and end
    func = getattr(TFNodesAST, curNodeOp[1:-1])
    (assignedVarAST, curAST) = func(graph, curNode,
                                    dictNodeNameToOutVarStr, extraNodeInfoDict)
    return (assignedVarAST, curAST)

# Takes the graph DS and outputs IR in SeeDot for the same


def generateIRCode(graph, extraInfoDict):
    program = None
    innerMostLetASTNode = None
    dictNodeNameToOutVarStr = {}
    outVarCt = 0
    outVarPrefix = "J"
    mtdAST = MtdAST()
    for curNode in graph.getAllNodesRef():
        for curInp in curNode.getInputsRef():
            # Consequence of topological sorting of the TF graph
            assert(curInp in dictNodeNameToOutVarStr)
        (assignedVarAST, curAst) = generateASTForNode(
            graph, curNode, dictNodeNameToOutVarStr, extraInfoDict)

        mtdForCurAST = {AST.ASTNode.mtdKeyTFOpName: curNode.getOp()[1:-1],
                        AST.ASTNode.mtdKeyTFNodeName: curNode.getName()[1:-1]}

        if (curAst is None):
            dictNodeNameToOutVarStr[curNode.getName()] = None
            continue
        curOutVarStr = outVarPrefix + str(outVarCt)
        curOutVarAstNode = (
            assignedVarAST if assignedVarAST else AST.ID(curOutVarStr))
        if program:
            assert(type(innerMostLetASTNode) is AST.Let)
            newNode = AST.Let(curOutVarAstNode, curAst, curOutVarAstNode)
            mtdAST.visit(newNode, mtdForCurAST)
            innerMostLetASTNode.expr = newNode
            innerMostLetASTNode = newNode
        else:
            innerMostLetASTNode = AST.Let(
                AST.ID(curOutVarStr), curAst, curOutVarAstNode)
            mtdAST.visit(innerMostLetASTNode, mtdForCurAST)
            innerMostLetASTNode.depth = 0
            program = innerMostLetASTNode
        dictNodeNameToOutVarStr[curNode.getName()] = curOutVarStr
        outVarCt += 1
    return (program, dictNodeNameToOutVarStr)


def countUniqueOps(graph):
    allOps = []
    for curNode in graph.getAllNodesRef():
        if (curNode.getOp() not in allOps):
            allOps.append(curNode.getOp())
    print("allOps.ct = ", len(allOps))
    gradientDesOps = []
    for curNode in graph.getAllNodesRef():
        if (curNode.getName().startswith("\"gradient_descent_optimizer")) and (curNode.getOp() not in gradientDesOps):
            gradientDesOps.append(curNode.getOp())
    print("allOps ct for gradient descent optimiser = ", len(gradientDesOps))
    return allOps


def readSizeInfo(fileName):
    allLines = None
    with open(fileName) as f:
        allLines = f.readlines()
    sizeInfo = {}
    for line in allLines:
        tokens = line.split()
        # assert(len(tokens) > 1) # Nodes with no size info are not getting outputted right now
        nodeName = tokens[0]
        tokens = tokens[1:]
        nodeOPSize = []
        if (not tokens):
            nodeOPSize = [1]
        else:
            for curDimStr in tokens:
                if (curDimStr == ''):
                    continue
                nodeOPSize.append(int(curDimStr))
        sizeInfo[nodeName] = nodeOPSize
    return sizeInfo

# Since later on in the pipeline, the placeholder nodes which come up as cin statements
# 	are to be excluded from the timing calculation, output all such PlaceHolder nodes together first.
#	This doesn't violate the topological ordering because all such PlaceHolder nodes are leaf nodes
# 	in the graph.


def prefixAllPlaceHolderNodes(graph):
    allNodes = graph.getAllNodesRef()
    placeHolderNodes = []
    remNodes = []
    for curNode in allNodes:
        if (curNode.getOp() == "\"Placeholder\"" or curNode.getOp() == "\"VariableV2\""):
            # Assert this is indeed a leaf node
            assert(len(curNode.getInputsRef()) == 0)
            placeHolderNodes.append(curNode)
        else:
            remNodes.append(curNode)
    graph.setNodesList(placeHolderNodes + remNodes)


def main():
    sys.setrecursionlimit(5000)
    # First read the graph file
    # if (len(sys.argv) < 2):
    #	print("FolderName unspecified.", file=sys.stderr)
    #	exit(1)

    #folderName = sys.argv[2]
    graphFileName = os.path.join('graphDef.txt')
    graph = Graph.Graph()
    with open(graphFileName) as file:
        graph.readFromFilePointer(file)

    # # Read the sizeInfo also
    sizeInfoFileName = os.path.join('sizeInfo.txt')
    sizeInfo = readSizeInfo(sizeInfoFileName)

    # Place all PlaceHolder nodes together at the beginning
    prefixAllPlaceHolderNodes(graph)

    # Re-format the input names of nodes
    for curNode in graph.getAllNodesRef():
        inputsRef = curNode.getInputsRef()
        for i, curInput in enumerate(inputsRef):
            # TODO for training : below is not correct
            # if (curInput.endswith(':1"')):
            # 	inputsRef[i] = curInput.split(':1')[0] + '"'
            if (curInput.startswith('"^')):
                # My hypothesis from empirical observation is that inputs which have '^' ahead of the node name
                #	denote control flow dependency and not data dependency.
                #	For all purposes for this compilation, control and data dependency is considered same.
                inputsRef[i] = '"' + curInput.split('^')[-1]

    # Create extra info dict
    # Format : (sizeInfo)
    extraInfoDict = {}
    for k, v in sizeInfo.items():
        extraInfoDict["\"" + k + "\""] = (v,)
    for curNode in graph.getAllNodesRef():
        if (curNode.getName() not in extraInfoDict):
            extraInfoDict[curNode.getName()] = (None,)

    print("Generating code from TF graph def : ", graphFileName, " ...")
    (program, dictNodeNameToOutVarStr) = generateIRCode(graph, extraInfoDict)
    #printAST = PrintAST()
    # printAST.visit(program)

    return program
    #print("SeeDot AST generation done. Pickling the AST.")
    # with open('astOutput.pkl', 'wb') as f:
    #	pickle.dump(program, f)

#    xx1 = countUniqueOps(graph)
#    filename = "./fileGraphDef_LSTM"
#    graph1 = Graph.Graph()
#    with open(filename) as file:
#        graph1.readFromFilePointer(file)
#    xx2 = countUniqueOps(graph1)


if __name__ == "__main__":
    main()
