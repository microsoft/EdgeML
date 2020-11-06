import sys
import enum

# Util functions


def errIfTokensNotMinLen(tokens, minlen, lineNum, entity):
    errCond = (len(tokens) < minlen)
    if (errCond):
        print("Less than expected number of tokens found while parsing",
              entity, " at line =", lineNum, file=sys.stderr)
    return errCond


class DataTypeEnum(enum.Enum):
    DT_INVALID = 0
    DT_FLOAT = 1
    DT_BOOL = 2
    DT_INT32 = 3
    DT_INT64 = 4

    def Parse(str):
        if (str == "DT_FLOAT"):
            return DataTypeEnum.DT_FLOAT
        elif (str == "DT_BOOL"):
            return DataTypeEnum.DT_BOOL
        elif (str == "DT_INT32"):
            return DataTypeEnum.DT_INT32
        elif (str == "DT_INT64"):
            return DataTypeEnum.DT_INT64
        else:
            return DataTypeEnum.DT_INVALID

    # TODO : The size given below is for c++ .. for python the sizes are different
    def Size(dt):
        if (dt == DataTypeEnum.DT_INVALID):
            return 0
        elif (dt == DataTypeEnum.DT_FLOAT):
            return 4
        elif (dt == DataTypeEnum.DT_BOOL):
            return 1
        elif (dt == DataTypeEnum.DT_INT32):
            return 4
        elif (dt == DataTypeEnum.DT_INT64):
            return 8
        else:
            raise Exception('Invalid dataTypeEnum value.')


class Shape:
    def __init__(self, dimList=None):
        if (dimList is None):
            self.__dimList = []
        else:
            self.__dimList = dimList
        self.__unknownRank = False

    def getNumElements(self):
        if self.__dimList:
            #list is non-empty
            ans = 1
            for curDim in self.__dimList:
                ans *= curDim
            return ans
        else:
            #list is empty
            return 0

    def getRank(self):
        return len(self.__dimList)

    def getDimRef(self):
        return self.__dimList

    def shapeUnknown(self):
        for curDim in self.__dimList:
            if (curDim == -1):
                return True
        return False

    def equalButOneDim(self, other, ignore_dim):
        if(self.__unknownRank != other.__unknownRank):
            return False
        if(self.__unknownRank == other.__unknownRank and self.__unknownRank == True):
            return True
        if (len(self.__dimList) != len(other.__dimList)):
            return False
        for i, curDim in enumerate(self.__dimList):
            if (i != ignore_dim and curDim != other.__dimList[i]):
                return False
        return True

    def __eq__(self, other):
        if (self.__unknownRank != other.__unknownRank):
            return False
        if ((self.__unknownRank == other.__unknownRank) and (self.__unknownRank == True)):
            return True
        if (len(self.__dimList) != len(other.__dimList)):
            return False
        for i, dimVal in enumerate(self.__dimList):
            if (other.__dimList[i] != dimVal):
                return False
        return True

    def __getitem__(self, index):
        return self.__dimList[index]

    def readFromFilePointer(self, fileP, cnt):
        line = fileP.readline()
        cnt += 1
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "Shape")):
                return (False, cnt)
            curToken = tokens[0]
            if (curToken == "}"):
                return (True, cnt)
            elif (curToken == "dim"):
                nline = fileP.readline()
                cnt += 1
                nlineTokens = nline.split()
                if (nlineTokens[0] == "}"):
                    return (True, cnt)
                elif (len(nlineTokens) != 2 or nlineTokens[0] != "size:"):
                    print("Error while parsing dim in shape at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)

                nnline = fileP.readline()
                cnt += 1
                nnlineTokens = nnline.split()
                if (len(nnlineTokens) != 1 or nnlineTokens[0] != "}"):
                    print("Error while parsing dim in shape at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)

                self.__dimList.append(int(nlineTokens[1]))
            elif (curToken == "unknown_rank:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Shape")):
                    return (False, cnt)
                self.__unknownRank = bool(tokens[1])
            else:
                print("Unknown token found while parsing shape at line =",
                      cnt, ", token =", curToken, file=sys.stderr)
                return (False, cnt)
            line = fileP.readline()
            cnt += 1
        return (False, cnt)

    def print(self):
        if (self.__unknownRank):
            print("Unknown rank")
        else:
            print("Size:", ",".join(list(map(str, self.__dimList))))


class Tensor:
    # TODO : Not impelemented operator Shape() from the c++ implementation.

    def __init__(self):
        # In the input, either tensor content in the form of a binary string is provided
        #   or an int/float/bool value is provided and the tensor shape is provided
        #   The corresponding C++ code converts the int/float/bool array into byte array.
        #   Right now I don't see any use of doing this in python implementation.
        #   So, for now, either __valArr will be non-null or __tensorBytes.
        #   TODO : If need arises, change everything to byte array.
        self.__totalSize = None
        self.__dtype = None
        self.__tensorShape = None
        self.__tensorContentInput = None
        self.__tensorBytes = None
        self.__valInput = None
        self.__valArr = None

    def getShapeRef(self):
        return self.__tensorShape

    def __convToBytes(self):
        numElements = self.__tensorShape.getNumElements()

        # TODO : The totalSize calculated below seems inaccurate because empty list in python is itself 64 bytes
        #       Meaning the actual size on a python system will be much more.

        self.__totalSize = DataTypeEnum.Size(self.__dtype)
        self.__totalSize *= numElements

        if ((self.__dtype == DataTypeEnum.DT_BOOL) and (self.__valInput is not None)):
            self.__valArr = [self.__valInput]*numElements
        elif ((self.__dtype == DataTypeEnum.DT_FLOAT) and (self.__valInput is not None)):
            self.__valArr = [self.__valInput]*numElements
        elif ((self.__dtype == DataTypeEnum.DT_INT32 or self.__dtype == DataTypeEnum.DT_INT64)
                and (self.__valInput is not None)):
            self.__valArr = [self.__valInput]*numElements
        else:
            # By virtue of how this function is called from below
            assert(self.__tensorContentInput)
            # Parse the tensorcontent and fill the tensorbytes

            self.__tensorBytes = bytearray(self.__totalSize)

            byteArrIdx = 0
            tsCtnIdx = 0
            while(self.__tensorContentInput[tsCtnIdx] != '\"'):
                tsCtnIdx += 1
            tsCtnIdx += 1
            while(tsCtnIdx + 1 < len(self.__tensorContentInput)):
                if (self.__tensorContentInput[tsCtnIdx] != '\\'):
                    self.__tensorBytes[byteArrIdx] = ord(
                        self.__tensorContentInput[tsCtnIdx])
                else:
                    if (self.__tensorContentInput[tsCtnIdx+1] == 'n'):
                        self.__tensorBytes[byteArrIdx] = 10
                        tsCtnIdx += 1
                    else:
                        self.__tensorBytes[byteArrIdx] = (ord(self.__tensorContentInput[tsCtnIdx+1])-ord('0'))*64 + (ord(
                            self.__tensorContentInput[tsCtnIdx+2])-ord('0'))*8 + (ord(self.__tensorContentInput[tsCtnIdx+3])-ord('0'))
                        tsCtnIdx += 3
                byteArrIdx += 1
                tsCtnIdx += 1

    def readFromFilePointer(self, fileP, cnt):
        line = fileP.readline()
        cnt += 1
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "Tensor")):
                return (False, cnt)
            curToken = tokens[0]
            if (curToken == "}"):
                return (True, cnt)
            elif (curToken == "tensor_shape"):
                sh = Shape()
                (noParseErr, cnt) = sh.readFromFilePointer(fileP, cnt)
                if not(noParseErr):
                    print(
                        "Error in reading shape while parsing tensor at line =", cnt, file=sys.stderr)
                    return (False, cnt)
                self.__tensorShape = sh
                if (len(self.__tensorShape.getDimRef()) == 0):
                    self.__totalSize = 0
            elif (curToken == "dtype:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Tensor")):
                    return (False, cnt)
                dtype = DataTypeEnum.Parse(tokens[1])
                if (dtype == DataTypeEnum.DT_INVALID):
                    print(
                        "Unknown dtype found while parsing Tensor at line =", cnt, file=sys.stderr)
                    return (False, cnt)
                else:
                    self.__dtype = dtype
            elif (curToken == "tensor_content:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Tensor")):
                    return (False, cnt)
                self.__tensorContentInput = tokens[1]
                self.__convToBytes()
            elif (curToken == "float_val:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Tensor")):
                    return (False, cnt)
                self.__valInput = float(tokens[1])
                self.__convToBytes()
            elif (curToken == "bool_val:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Tensor")):
                    return (False, cnt)
                self.__valInput = bool(tokens[1])
                self.__convToBytes()
            elif (curToken == "int_val:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Tensor")):
                    return (False, cnt)
                self.__valInput = int(tokens[1])
                self.__convToBytes()
            else:
                print("Unknown token found while parsing Tensor at line =",
                      cnt, ", token =", curToken, file=sys.stderr)
                return (False, cnt)
            line = fileP.readline()
            cnt += 1
        return (False, cnt)

    def print(self):
        print("DType:", self.__dtype)
        print("Shape: ", end="")
        self.__tensorShape.print()
        print("Content:", self.__tensorContentInput)
        print("ShapeRank:", self.__tensorShape.getRank())
        print("TotalSizeBytes:", self.__totalSize)
        if (self.__tensorBytes):
            print("ActualContentBytes:", self.__tensorBytes)
        else:
            print("ValArr:", self.__valArr)

    def getConstantVal(self):
        return self.__valInput

    def getContentAsValArr(self):
        # This will try and return an array of values (even when tensorContent is given as array of bytes)
        if self.__valArr:
            pass
        else:
            # Convert tensorBytes into array of values
            numOfElements = self.__tensorShape.getNumElements()
            numOfBytesPerVal = None
            if self.__dtype == DataTypeEnum.DT_INT32:
                numOfBytesPerVal = 4
            else:
                # Right now the CNN tensorflow benchmark i am dealing with only has int32 case when tensorContents are given as bytes.
                # If in future, we encounter this case for float/bool, deal with it accordingly here
                # Plus, also from empirical observation, byteorder is little and its a signed value for ints.
                # Figure out for others when the time comes.
                print(self.__dtype)
                assert False
            it = 0
            returnArr = []
            while(it <= len(self.__tensorBytes)-1):
                curInt = int.from_bytes(
                    self.__tensorBytes[it:it+numOfBytesPerVal], byteorder='little', signed=True)
                returnArr.append(curInt)
                it += numOfBytesPerVal
            self.__valArr = returnArr
        return self.__valArr


class MultiValue:
    def __init__(self):
        self.__valStrLi = []
        self.__valIntLi = []
        self.__valFloatLi = []
        self.__valBoolLi = []

    def readFromFilePointer(self, fileP, cnt):
        line = fileP.readline()
        cnt += 1
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "Multivalue")):
                return (False, cnt)
            curToken = tokens[0]
            if (curToken == "}"):
                return (True, cnt)
            elif (curToken == "s:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Multivalue")):
                    return (False, cnt)
                self.__valStrLi.append(tokens[1])
            elif (curToken == "f:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Multivalue")):
                    return (False, cnt)
                self.__valFloatLi.append(float(tokens[1]))
            elif (curToken == "i:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Multivalue")):
                    return (False, cnt)
                self.__valIntLi.append(int(tokens[1]))
            elif (curToken == "b:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Multivalue")):
                    return (False, cnt)
                self.__valBoolLi.append(bool(tokens[1]))
            else:
                print("Unknown token found while parsing Mutlivalue, line =",
                      cnt, ", Token =", curToken, file=sys.stderr)
                return (False, cnt)
            line = fileP.readline()
            cnt += 1
        return (False, cnt)

    def print(self):
        print("sses:", ",".join(self.__valStrLi))
        print("is:", ",".join(list(map(str, self.__valIntLi))))
        print("fs:", ",".join(list(map(str, self.__valFloatLi))))
        print("bs:", ",".join(list(map(str, self.__valBoolLi))))

    def getILi(self):
        if len(self.__valIntLi) > 0:
            assert(all((type(x) is int) for x in self.__valIntLi))
        return self.__valIntLi


class Value:
    def __init__(self):
        self.__val = None

    def readFromFilePointer(self, fileP, cnt):
        line = fileP.readline()
        cnt += 1
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "Value")):
                return (False, cnt)
            curToken = tokens[0]
            if (curToken == "}"):
                return (True, cnt)
            elif (curToken == "s:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Value")):
                    return (False, cnt)
                self.__val = tokens[1]
            elif (curToken == "i:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Value")):
                    return (False, cnt)
                self.__val = int(tokens[1])
            elif (curToken == "f:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Value")):
                    return (False, cnt)
                self.__val = float(tokens[1])
            elif (curToken == "b:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Value")):
                    return (False, cnt)
                self.__val = bool(tokens[1] == "true")
            elif (curToken == "type:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "Value")):
                    return (False, cnt)
                dtype = DataTypeEnum.Parse(tokens[1])
                if (dtype == DataTypeEnum.DT_INVALID):
                    print("Invalid dtype found while parsing Value at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)
                else:
                    self.__val = dtype
            elif (curToken == "shape"):
                sh = Shape()
                (noParseError, cnt) = sh.readFromFilePointer(fileP, cnt)
                if (not(noParseError)):
                    print("Error in parsing Value at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)
                self.__val = sh
            elif (curToken == "list"):
                mv = MultiValue()
                (noParseError, cnt) = mv.readFromFilePointer(fileP, cnt)
                if (not(noParseError)):
                    print("Error in parsing Value at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)
                self.__val = mv
            elif (curToken == "tensor"):
                ts = Tensor()
                (noParseError, cnt) = ts.readFromFilePointer(fileP, cnt)
                if (not(noParseError)):
                    print("Error in parsing Value at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)
                self.__val = ts
            else:
                print("Unknown token while parsing Value at line =",
                      cnt, ", token =", curToken, file=sys.stderr)
                return (False, cnt)
            line = fileP.readline()
            cnt += 1
        return (False, cnt)

    def print(self):
        if (type(self.__val) is str):
            print("s:", self.__val)
        elif (type(self.__val) is int):
            print("i:", self.__val)
        elif (type(self.__val) is float):
            print("f:", self.__val)
        elif (type(self.__val) is bool):
            print("b:", self.__val)
        elif (type(self.__val) is DataTypeEnum):
            print("Type:", self.__val)
        elif (type(self.__val) is Shape):
            print("Shape: ", end="")
            self.__val.print()
        elif (type(self.__val) is Tensor):
            print("Tensor: ", end="")
            self.__val.print()
        elif (type(self.__val) is MultiValue):
            print("List: ", end="")
            self.__val.print()
        else:
            assert(False)

    def getS(self):
        assert(type(self.__val) is str)
        return self.__val

    def getI(self):
        assert(type(self.__val) is int)
        return self.__val

    def getF(self):
        assert(type(self.__val) is float)
        return self.__val

    def getB(self):
        assert(type(self.__val) is bool)
        return self.__val

    def getDataType(self):
        assert(type(self.__val) is DataTypeEnum)
        return self.__val

    def getShape(self):
        assert(type(self.__val) is Shape)
        return self.__val

    def getTensor(self):
        assert(type(self.__val) is Tensor)
        return self.__val

    def getList(self):
        assert(type(self.__val) is MultiValue)
        return self.__val


class Node:
    def __init__(self):
        self.__name = ""  # Name of node
        self.__op = ""  # Name of operation carried out by node
        self.__inputs = []  # List of all inputs to the current node
        # Map of (attrName, Value) of all attributes for the current node
        self.__attr = {}

    def getName(self):
        return self.__name

    def getOp(self):
        return self.__op

    def getInputsRef(self):
        return self.__inputs

    def getAttrMapRef(self):
        return self.__attr

    def readAttrFromFilePointer(self, fileP, cnt):
        line = fileP.readline()
        cnt += 1
        keyStr = None
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "attr from node")):
                return (False, cnt)
            curToken = tokens[0]
            if (curToken == "}"):
                return (True, cnt)
            elif (curToken == "key:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "attr from node")):
                    return (False, cnt)
                if (keyStr):
                    # keyStr is already non-None .. there is then probably some error
                    print(
                        "Too many keys found while parsing attr for node at line =", cnt, file=sys.stderr)
                    return (False, cnt)
                keyStr = tokens[1]
            elif (curToken == "value"):
                curVal = Value()
                (noParseError, cnt) = curVal.readFromFilePointer(fileP, cnt)
                if not(noParseError):
                    print(
                        "Error while parsing value of attr for node at line =", cnt, file=sys.stderr)
                    return (False, cnt)
                if not(keyStr):
                    print(
                        "Value found - but no key found for attr in node at line =", cnt, file=sys.stderr)
                    return (False, cnt)
                self.__attr[keyStr] = curVal
            else:
                print("Unrecognized token found while parsing attribute for node at line =",
                      cnt, ", token =", curToken, file=sys.stderr)
                return (False, cnt)
            line = fileP.readline()
            cnt += 1
        return (False, cnt)

    def readFromFilePointer(self, fileP, cnt):
        line = fileP.readline()
        cnt += 1
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "node")):
                return (False, cnt)
            curToken = tokens[0]
            if (curToken == "}"):
                return (True, cnt)
            elif (curToken == "name:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "node")):
                    return (False, cnt)
                self.__name = tokens[1]
            elif (curToken == "op:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "node")):
                    return (False, cnt)
                self.__op = tokens[1]
            elif (curToken == "input:"):
                if (errIfTokensNotMinLen(tokens, 2, cnt, "node")):
                    return (False, cnt)
                self.__inputs.append(tokens[1])
            elif (curToken == "attr"):
                (noParseError, cnt) = self.readAttrFromFilePointer(fileP, cnt)
                if (not(noParseError)):
                    print("Error parsing node data at line =",
                          cnt, file=sys.stderr)
                    return (False, cnt)
            else:
                print("Unrecognized token found while parsing node data at line =",
                      cnt, ", token =", curToken, file=sys.stderr)
                return (False, cnt)
            line = fileP.readline()
            cnt += 1
        return (False, cnt)

    def print(self):
        print("NODE::")
        print(self.__name, ",", self.__op)
        print("Inputs:")
        for inp in self.__inputs:
            print(inp)
        for attrKey, attrVal in self.__attr.items():
            print("Attr:", attrKey)
            attrVal.print()


class Graph:
    def __init__(self):
        self.__Nodes = {}  # Map of (op, Node)
        # Sequential list of nodes in the order in which its specified in graph_def.
        self.__NodesLi = []

    def getAllNodesRef(self):
        return self.__NodesLi

    def setNodesList(self, nodesLi):
        self.__NodesLi = nodesLi

    def readFromFilePointer(self, fileP):
        line = fileP.readline()
        cnt = 1
        while line:
            tokens = line.split()
            if (errIfTokensNotMinLen(tokens, 1, cnt, "graph")):
                return False
            curToken = tokens[0]
            if (curToken == "node"):
                curNode = Node()
                (noPaseError, cnt) = curNode.readFromFilePointer(fileP, cnt)
                if (noPaseError):
                    self.__Nodes[curNode.getOp()] = curNode
                    self.__NodesLi.append(curNode)
                else:
                    print("Error parsing graph dump for node at line =",
                          cnt, file=sys.stderr)
                    return False
            elif (curToken == "}"):
                # CurNode ended
                pass
            elif (curToken == "versions"):
                print("Versions node found. Ignoring remainder graph. Line =",
                      cnt, file=sys.stderr)
                return True
            else:
                print("Unrecognized token in graph dump at line =",
                      cnt, ", token =", curToken, file=sys.stderr)
                return False
            line = fileP.readline()
            cnt += 1
        print("Graph parsing successful.")
        return True

    def __getitem__(self, opName):
        return self.__Nodes[opName]

    def print(self):
        for _, curNode in self.__Nodes.items():
            curNode.print()
