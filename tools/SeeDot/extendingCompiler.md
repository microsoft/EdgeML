# Extending The Compiler

SeeDot translates the given input code into a sequence of function calls. The functions are already implemented in a library. What if the model being implemented needs a function which does not already implemented in the library? We show how to add the convolution operator.

**Note**: We only show how to add a new node for X86 architecture. Adding the node for Arduino is omitted for brevity, but follows very closely with the X86 guidelines.

### STEP 1: Operator description

We present an implementation of convolution operation which supports padding, strides, dilations, as well as groups (for depthwise-separable convolutions) which is similar to what is provided by pytorch [here](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d).

Suppose our syntax looks like the following:
``` ocaml
    let A = <some N*H*W*C tensor>
    let B = <some G*FH*FW*C*Cout tensor>
    let C = conv2d(A, B, {s 1 1}, {p 0 0 1 1}, {d 1 1}, {g 2})
```
In this syntax, `s` represents the 2 stride parameters (vertical, horizontal), `p` represents the 4 padding parameters (up, down, left, right), `d` represents the 4 dilation parameters (vertical, horizontal), `g` represents the number of groups. For details please refer to the pytorch documentation of `conv2d` given above.

### STEP 2: Adding the grammar in the file

1. The grammar is in the file `SeeDot\seedot\compiler\antlr\seedot.g4`. Add the following (feel free to change symbols if preferred):
``` bash
    | Conv2d '(' expr ',' expr ','
    '{s' IntConst IntConst '}' ','
    '{p' IntConst IntConst IntConst IntConst'}' ','
    '{d' IntConst IntConst '}' ','
    '{g'IntConst'}' ')' # convolution
```
to the rules for `expr`. The name after the # (convolution) is the name which would be generated for this operator when we generate the parser. (Watch out for methods named `visitConvolution` below).

2. After updating the grammar file, we need to generate a new parser. Navigate to `SeeDot\seedot\lib\` in a terminal and execute the following command:
``` bash
    java -jar .\antlr-4.7-complete.jar ..\compiler\antlr\seedot.g4 -visitor -Dlanguage=Python3 -o kl
```
This will generate the new parser and associated files in the folder `SeeDot\seedot\lib\compiler\antlr\`.

3. Within the folder `SeeDot\seedot\lib\compiler\antlr\` there will be 6 files. Delete `seedotListener.py` (it is not needed) and copy the rest of the files to the directory `SeeDot\seedot\compiler\antlr\` (there already will be 5 files of the same name which you must overwrite). After this SeeDot's parser will be updated.

### STEP 3: Adding new nodes in the AST

Since we are adding a new rule for the expr, we will need to add a new node for our convolution operation to the abstract syntax tree (AST).
1. Go to the file `SeeDot\seedot\compiler\ast\ast.py`.
Here, we will add a node which captures all the relevant information for the convolution node. We have discussed above that the node has an input image, an input filter, properties called stride, padding, dilation, groups. So we add the following class to the end of the file:
``` python
    class Convolution(ASTNode):
        
        def __init__(self, expr1, expr2, stride, padding, dilation, groups):
            super().__init__()
            self.expr1 = expr1
            self.expr2 = expr2
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
```
2. We also need a mechanism to construct an object of the class `Convolution`. Navigate to `SeeDot\seedot\compiler\ast\astBuilder.py`, and add the following method as a member of the class `ASTBuilder`:
``` python
    def visitConvolution(self, ctx: SeeDotParser.ConvolutionContext):
        expr1 = self.visit(ctx.expr(0))
        expr2 = self.visit(ctx.expr(1))
        stride = [int(ctx.IntConst(i).getText()) for i in range(0, 2)]
        padding = [int(ctx.IntConst(i).getText()) for i in range(2, 6)]
        dilation = [int(ctx.IntConst(i).getText()) for i in range(6, 8)]
        groups = int(ctx.IntConst(8).getText())
        return AST.Convolution(expr1, expr2, stride, padding, dilation, groups)
```
We explain the above builder method. Take a look at the addition to the grammar file in **STEP 1**. In a convolution node, the first two sub-expressions signify the two inputs. Since they are sub-expressions within our convolution expression, we must recurse down both of them one after the other. `ctx.expr(0)` refers to first sub-expression, `ctx.expr(1)` refers to the second sub-expression.
After the first two sub-expressions, we have 2 integers (0, 1) for the stride parameter, 4 integers (2, 3, 4, 5) for padding, 2 integers (6, 7) for dilation, and 1 integer (8) for groups. All of these numbers are read by the parser in the same order, and we extract the numbers. Thus the `ctx.expr()`, `ctx.IntConst(i).getText()` etc. are pre-generated from the parser in **STEP 2**. After extracting all parameters, we simply construct the `Convolution` class and return.

3. Navigate to `SeeDot\seedot\compiler\ast\astVisitor.py`. The `ASTVisitor` class is inherited by the type checker, IR generator etc. In the `visit()` method, before the else branch, add the following:
``` python
    elif isinstance(node, AST.Convolution):
        return self.visitConvolution(node)
```
4. Edit some boilerplate code. In `SeeDot\seedot\compiler\ast\mtdAST.py`, within the `MtdAST` class, add the following method:
``` python
    def visitConvolution(self, node: AST.Convolution, mtd: dict):
        node.metadata.update(mtd)
        self.visit(node.expr1, mtd)
        self.visit(node.expr2, mtd)
```
Include all the sub-expressions of the node which is being added, but no need to add `int` constants (which are not recursively explored).

5. In `SeeDot\seedot\compiler\ast\printAST.py`, within the `PrintAST` class, add the following method:
``` python
    def visitConvolution(self, node: AST.Convolution):
        node.expr1.printLevel = node.expr2.printLevel = node.printLevel + 1
        print(indent * node.printLevel, "conv(", )
        self.visit(node.expr1)
        self.visit(node.expr2)
        print(",", node.stride, ',', node.padding, ',', node.dilation, ',',node.groups, ')')
```
This is for pretty printing. Feel free to edit if not pretty enough.

### STEP 4: Type Checking

Navigate to `SeeDot\seedot\compiler\type.py` and add the following code as a member method of `Type` class:
``` python
def visitConvolution(self, node: ast.Convolution):
    node.expr1.gamma = dict(node.gamma)
    eType = self.visit(node.expr1)

    assert eType.dim == 4
    [n, h, w, cin] = eType.shape

    node.expr2.gamma = dict(node.gamma)
    fType = self.visit(node.expr2)

    assert fType.dim == 5
    [g, hf, wf, cin_, cout] = fType.shape

    assert cin_ * g == cin
    assert g == node.groups

    assert hf % 2 == wf % 2 == 1, "Odd filter sizes supported"

    for i in range(0,4):
        assert node.padding[i] >= 0, "Padding cannot be negative"
        assert node.stride[0] > 0 and node.stride[1] > 0, "Stride must be positive"
        assert node.dilation[0] > 0 and node.dilation[1] > 0, "Dilation must be positive"

    hout = (h + node.padding[0] + node.padding[1] - node.dilation[0] * (hf - 1) - 1) // node.stride[0] + 1
    wout = (w + node.padding[2] + node.padding[3] - node.dilation[1] * (wf - 1) - 1) // node.stride[1] + 1
    shape = [n, hout, wout, g * cout]

    node.type = Tensor(shape)
    return node.type
 ```
This function checks whether the types of the inputs are compatible with the outputs, whether there are illegal values and so on.
For example, the following are the checks made for convolution:
1. The input image should be of dimension 4 corresponding to (batch, number of rows, number of columns, channels), and the filter should match the number of channels.
2. Our implementation does not support even filter sizes, so we can make a check here to only allow odd filter sizes or throw an exception.
3. The values for padding should not be negative, neither should dilations be negative
The method also computes the type of the output node, given the input (look at the assignment `node.type = Tensor(shape)`, which is returned).
For other operators, different checks may be there. For example for addition, the dimensions of both inputs should match and the output should also have the same dimension as inputs.

### STEP 5: Implementing the operator
**Note**: *This part can be tricky as it involves a good understanding of fixed point arithmetic and how to best tune the hyperparameters which are introduced because of fixed point.
It is recommended to go through MatAdd, MatMul, Exponentiation function codes to get an idea of how to write fixed point operations.*

1. We first add an implementation of the operator in floating point. We navigate to `SeeDot\seedot\Predictor\library_float.h` and add the following:
``` cpp
void Convolution(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2);
```
2. We navigate to `SeeDot\seedot\Predictor\library_float.cpp` and add the following:
``` cpp
void Convolution(float *A, const float *B, float *C, float *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2) {
    MYITE HOffsetL = HDL*(HF/2) - HPADL;
    MYITE WOffsetL = WDL*(WF/2) - WPADL;
    MYITE HOffsetR = HDL*(HF/2) - HPADR;
    MYITE WOffsetR = WDL*(WF/2) - WPADR;

    for (MYITE n = 0; n < N; n++) {
        for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
            for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
                for (MYITE g = 0; g < G; g++) {
                    for (MYITE co = 0; co < COUTF; co++) {
                        MYITE counter = 0;

                        for (MYITE hf = -(HF/2); hf <= HF/2; hf++) {
                            for (MYITE wf = -(WF/2); wf <= WF/2; wf++) {
                                for (MYITE ci = 0; ci < CINF; ci++) {
                                    float a = (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
                                    float b = B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF + COUTF + ci * COUTF + co];
                                    tmp[counter] = a * b;
                                    counter++;
                                }
                            }
                        }

                        MYITE totalEle = HF * WF * CINF;
                        MYITE count = HF * WF * CINF, depth = 0;

                        bool shr = true;
                        while (depth < (H1 + H2)) {
                            if (depth >= H1) {
                                shr = false;
                            }
                            for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
                                float sum;
                                if (p < (count >> 1)) {
                                    sum = tmp[2 * p] + tmp[(2 * p) + 1];
                                } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
                                    sum = tmp[2 * p];
                                } else {
                                    sum = 0;
                                }

                                if (shr) {
                                    tmp[p] = sum;
                                } else {
                                    tmp[p] = sum;
                                }
                            }

                            count = (count + 1) >> 1;
                            depth++;
                        }
                        C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = tmp[0];
                    }
                }
            }
        }
    }
}
```
In most cases, this implementation itself can act as a reference for adding other operators. The code from `MYITE counter = 0;` to `MYITE totalEle = HF * WF * CINF` is the
standard convolution operation, and the while loop in the code is tree-sum addition (From SeeDot, Gopinath et al., PLDI 2019).

3. Navigate to `SeeDot\seedot\Predictor\library_fixed.h` and append the following:
``` cpp
template<class TypeA, class TypeB, class TypeTemp, class TypeC>
void Convolution(TypeA *A, const TypeB *B, TypeC *C, TypeTemp *tmp, MYINT N, MYINT H, MYINT W, MYINT CIN, MYINT HF, MYINT WF, MYINT CINF, MYINT COUTF, MYINT HOUT, MYINT WOUT, MYINT HPADL, MYINT HPADR, MYINT WPADL, MYINT WPADR, MYINT HSTR, MYINT WSTR, MYINT HDL, MYINT WDL, MYINT G, MYINT shrA, MYINT shrB, MYINT H1, MYINT H2, MYINT demote) {
    // Most parameters are self explanatory ones (parameters like inputs, outputs, sizes of inputs and outputs and filters). TypeA, TypeB, TypeTemp, TypeC can be int8_t, int16_t or int32_t, and represent the bitwidths for input image, input filter, temp buffer, output image.
    // shrA, shrB, H1, H2, demote are parameters pertaining to fixed point code:
    // shrA and shrB and demote are used for controlling scale during fixed point multiplication (Refer to Section 2.2 and Section 3 of the paper)
    // H1 and H2 are parameters of tree sum (the while loop in the function), which is used to sum up long vectors without losing significant bits
    MYITE HOffsetL = HDL*(HF/2) - HPADL;
    MYITE WOffsetL = WDL*(WF/2) - WPADL;
    MYITE HOffsetR = HDL*(HF/2) - HPADR;
    MYITE WOffsetR = WDL*(WF/2) - WPADR;

    for (MYITE n = 0; n < N; n++) {
        for (MYITE h = HOffsetL, hout = 0; h < H - HOffsetR; h += HSTR, hout++) {
            for (MYITE w = WOffsetL, wout = 0; w < W - WOffsetR; w += WSTR, wout++) {
                for (MYITE g = 0; g < G; g++) {
                    for (MYITE co = 0; co < COUTF; co++) {

                        MYITE counter = 0;
                        for (MYITE hf = -(HF/2); hf <= HF/2; hf++) {
                            for (MYITE wf = -(WF/2); wf <= WF/2; wf++) {
                                for (MYITE ci = 0; ci < CINF; ci++) {

                                    TypeTemp a = (TypeTemp) (((h + HDL * hf) < 0) || ((h + HDL * hf) >= H) || ((w + WDL * wf) < 0) || ((w + WDL * wf) >= W)) ? 0 : A[n * H * W * CIN + (h + HDL * hf) * W * CIN + (w + WDL * wf) * CIN + (ci + g * CINF)];
                                    TypeTemp b = (TypeTemp) B[g * HF * WF * CINF * COUTF + (hf + HF/2) * WF * CINF * COUTF + (wf + WF/2) * CINF * COUTF + ci * COUTF + co];
                                    tmp[counter] = a * b;
                                    counter++;
                                }
                            }
                        }

                        MYITE totalEle = HF * WF * CINF;
                        MYITE count = HF * WF * CINF, depth = 0;

                        bool shr = true;
                        while (depth < (H1 + H2)) {
                            if (depth >= H1) {
                                shr = false;
                            }
                            for (MYITE p = 0; p < (totalEle / 2 + 1); p++) {
                                TypeTemp sum;
                                if (p < (count >> 1)) {
                                    if (shr) {
                                        sum = tmp[2 * p] / 2 + tmp[(2 * p) + 1] / 2;
                                    } else {
                                        sum = tmp[2 * p] + tmp[(2 * p) + 1];
                                    }
                                } else if ((p == (count >> 1)) && ((count & 1) == 1)) {
                                    if (shr) {
                                        sum = tmp[2 * p] / 2;
                                    } else {
                                        sum = tmp[2 * p];
                                    }
                                } else {
                                    sum = 0;
                                }

                                tmp[p] = sum;
                            }

                            count = (count + 1) >> 1;
                            depth++;
                        }

                        C[n * HOUT * WOUT * (COUTF * G) + hout * WOUT * (COUTF * G) + wout * (COUTF * G) + (co + g * COUTF)] = Saturate<TypeC>(((tmp[0] / shrA) / shrB) / demote);
                    }
                }
            }
        }
    }
}
```

### STEP 6: Handling the operator in IR
**Note**: *This part can also be tricky as the user would need to know how to compute bit-widths, scales for the outputs given the input parameters. It is recommended to study MatAdd, MatMul, Exponentiation, TanH functions for a good understanding*

Navigate to `SeeDot\seedot\compiler\ir\irBuilder.py`, and add the following as a member method of the `IRBuilder` class:
``` Python
def visitConvolution(self, node: AST.Convolution):

    (prog_in_A, expr_in_A) = self.visit(node.expr1)
    (prog_in_B, expr_in_B) = self.visit(node.expr2)

# The above two are required to explore the AST of the input subexpression. If a node has n subexpressions, there would be n statements instead of the two above for each subexpression

    [expr_treeSum, expr_out] = self.getTempVars(2)

# In the above a new temporary variable will be assigned to both the output, and the temporary buffer used by the function

    [N, H, W, Cin] = node.expr1.type.shape
    [G, Hf, Wf, CinF, CoutF] = node.expr2.type.shape

 # The type checked inputs' dimensions are extracted from node, as out implementation expects these values. It may or may not be necessary to extract all such information

    type_treeSum = Type.Tensor([Hf * Wf * CinF])
    type_out = node.type

 # In the convolution operator, a temporary buffer is created whose type can be inferred from the inputs' types, but will not be encountered in the source code. Hence they are declared here

    bitwidth_in_A, scale_in_A = self.getBitwidthAndScale(expr_in_A.idf)
    bitwidth_in_B, scale_in_B = self.getBitwidthAndScale(expr_in_B.idf)
    if self.ddsEnabled:
        bitwidth_out, scale_out = self.getBitwidthAndScale(expr_out.idf)
        bitwidth_temp, scale_temp = self.getBitwidthAndScale(expr_out.idf, native=True)
    else:
        bitwidth_out = config.wordLength // 2 if expr_out.idf in self.demotedVarsList else config.wordLength
        scale_out, scale_temp = None, None
        bitwidth_temp = bitwidth_out

# ddsEnabled flag being true means that the output's scale is computed through profiling. if the flag is false, the scale would have to be computed manually. The getBitwidtAndScale method always returns the scale and the bitwidth assignment (It is known which all variables are in 8 bits before the IRBuilder class is invoked)

    intv_in_A, intv_in_B = (0, 0), (0, 0)
    intv_out = (0, 0)

    shr_A, shr_B, H1, H2, demote, scale_out = self.getShrTreeSumAndDemoteParamsForMul(bitwidth_in_A, scale_in_A, bitwidth_in_B, scale_in_B, bitwidth_temp, scale_temp, bitwidth_out, scale_out, Hf * Wf * CinF)

 # The helper method getShrTreeSumAndDemoteParamsForMul can be directly used to infer scales for multiplication outputs, and also compute the parameters which the fixed point implementation would expect for multiplication. There are many such helper methods, check out the node for MatAdd for addition.
 
    shr_A = self.formatShr(shr_A)
    shr_B = self.formatShr(shr_B)
 
    expr_in_A.inputVar = False
    expr_in_B.inputVar = False
    expr_out.inputVar = False
    expr_treeSum.inputVar = False
  
    if forFixed():
        self.varsForBitwidth[expr_treeSum.idf] = bitwidth_temp
 
    comment = IR.Comment('conv(%s, %s)' %(expr_in_A.idf,expr_in_B.idf), self.counter_inst+1)
    self.allDepths[self.counter_inst+1] = self.curDepth
 
    bitwidth_mul = self.getTempBitwidth(bitwidth_in_A, bitwidth_in_B, "mul")
    if self.vbwEnabled:
        self.varsForBitwidth[expr_treeSum.idf] = bitwidth_mul
 
# The temporary buffer variables are introduced by the compiler, and are not profiled by the input code. Their bitwidths are set using the values of bitwidths of the input

    argMap = {
        expr_in_A: "A",
        expr_in_B: "B",
        expr_out: "C",
        expr_treeSum: "tmp",
        IR.Int(N): "N",
        IR.Int(H): "H",
        IR.Int(W): "W",
        IR.Int(Cin): "CIN",
        IR.Int(Hf): "HF",
        IR.Int(Wf): "WF",
        IR.Int(CinF): "CINF",
        IR.Int(CoutF): "COUTF",
        IR.Int(type_out.shape[1]): "HOUT",
        IR.Int(type_out.shape[2]): "WOUT",
        IR.Int(node.padding[0]): "HPADL",
        IR.Int(node.padding[1]): "HPADR",
        IR.Int(node.padding[2]): "WPADL",
        IR.Int(node.padding[3]): "WPADR",
        IR.Int(node.stride[0]): "HSTR",
        IR.Int(node.stride[1]): "WSTR",
        IR.Int(node.dilation[0]): "HDL",
        IR.Int(node.dilation[1]): "WDL",
        IR.Int(G): "G",
        shr_A: "shrA",
        shr_B: "shrB",
        IR.Int(H1): "H1",
        IR.Int(H2): "H2"
    }

    if self.vbwEnabled:
        argMap[IR.Int(demote)] = "demote"

    if not self.vbwEnabled:
        funcCall = IR.FuncCall("Convolution", argMap) #, {expr_treeSum.idf: type_treeSum})
    else:
        funcCall = IR.FuncCall("Convolution" + ("<int%d_t, int%d_t, int%d_t, int%d_t>"%(bitwidth_in_A, bitwidth_in_B, bitwidth_mul, bitwidth_out)), argMap)
 
 # The argMap variable holds the arguments of the function call. Each key corresponds to one argument, the value is not required but is added for reference
 
    self.counter_inst += 1
    self.updateLiveRange([expr_in_A, expr_in_B, expr_out, expr_treeSum])
 
 # The above updateLiveRange method updates the live ranges of the variable being alive. Refer to Section 6 in the paper about Memory Management

    profile = IR.FuncCall("Profile4", {
        expr_out: "Var",
        IR.Int(N): "I",
        IR.Int(type_out.shape[1]): "J",
        IR.Int(type_out.shape[2]): "K",
        IR.Int(CoutF * G): "L",
        IR.String(expr_out): "VarName"
    })

    if forFloat():
        self.independentVars.append(expr_out.idf)

 # For convolution, the output variable is profiled from the data to infer it's scale. For any variable which needs to be profiled, the above function call is added (it gets added to the floating point code), If a variable's output scale is known (for example sigmoid's output is always bounded by +-1, so it need not be profiled and the output scale can be set directly)

    prog_conv = IR.Prog([comment, funcCall, profile] if forFloat() and self.ddsEnabled else [comment, funcCall])

    prog_out = IRUtil.concatPrograms(prog_in_A, prog_in_B, prog_conv)

# Update context for output variable
    self.varDeclarations[expr_out.idf] = type_out
    self.varScales[expr_out.idf] = scale_out
    self.varIntervals[expr_out.idf] = intv_out
    
    self.varDeclarations[expr_treeSum.idf] = type_treeSum
    self.varScales[expr_treeSum.idf] = scale_temp
    self.varIntervals[expr_treeSum.idf] = (0, 0)

# varDeclarations variable is to make a list of variables which need to be instantiated in the output code

    self.log.print(comment.msg)
    self.log.print("\tInput1: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_A.idf],) + self.varIntervals[expr_in_A.idf]))
    self.log.print("\tInput2: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_in_B.idf],) + self.varIntervals[expr_in_B.idf]))
    self.log.print("\tOutput: scale = %d, interval = [%d, %d]" % ((self.varScales[expr_out.idf],) + self.varIntervals[expr_out.idf]))

    return (prog_out, expr_out)
```

### STEP 7: Use the operator

After all these steps, one can simply use the operator in the input Shiftry code, and the compiler will handle the rest.
