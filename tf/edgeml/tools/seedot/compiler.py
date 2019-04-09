# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

from antlr4 import *
import argparse
import os

from edgeml.tools.seedot.antlr.seedotLexer import seedotLexer as SeeDotLexer
from edgeml.tools.seedot.antlr.seedotParser import seedotParser as SeeDotParser

import edgeml.tools.seedot.ast.ast as AST
import edgeml.tools.seedot.ast.astBuilder as ASTBuilder
from edgeml.tools.seedot.ast.printAST import PrintAST

from edgeml.tools.seedot.codegen.arduino import Arduino as ArduinoCodegen
from edgeml.tools.seedot.codegen.x86 import X86 as X86Codegen

from edgeml.tools.seedot.ir.irBuilder import IRBuilder
import edgeml.tools.seedot.ir.irUtil as IRUtil

from edgeml.tools.seedot.type import InferType
from edgeml.tools.seedot.util import *
from edgeml.tools.seedot.writer import Writer


class Compiler:

    def __init__(self, algo, target, inputFile, outputFile, profileLogFile, maxExpnt, numWorkers):
        if os.path.isfile(inputFile) == False:
            raise Exception("Input file doesn't exist")

        setAlgo(algo)
        setTarget(target)
        setNumWorkers(numWorkers)
        self.input = FileStream(inputFile)
        self.outputFile = outputFile
        setProfileLogFile(profileLogFile)
        setMaxExpnt(maxExpnt)

    def run(self):
        # Parse and generate CST for the input
        lexer = SeeDotLexer(self.input)
        tokens = CommonTokenStream(lexer)
        parser = SeeDotParser(tokens)
        tree = parser.expr()

        # Generate AST
        ast = ASTBuilder.ASTBuilder().visit(tree)

        # Pretty printing AST
        # PrintAST().visit(ast)

        # Perform type inference
        InferType().visit(ast)

        IRUtil.init()

        res, state = self.compile(ast)

        writer = Writer(self.outputFile)

        if forArduino():
            codegen = ArduinoCodegen(writer, *state)
        elif forX86():
            codegen = X86Codegen(writer, *state)
        else:
            assert False

        codegen.printAll(*res)

        writer.close()

    def compile(self, ast):
        if genFuncCalls():
            return self.genCodeWithFuncCalls(ast)
        else:
            return self.genCodeWithoutFuncCalls(ast)

    def genCodeWithFuncCalls(self, ast):

        compiler = IRBuilder()

        res = compiler.visit(ast)

        state = compiler.decls, compiler.scales, compiler.intvs, compiler.cnsts, compiler.expTables, compiler.globalVars

        return res, state

    def genCodeWithoutFuncCalls(self, ast):

        if forArduino() or forX86():
            compiler = ArduinoIRGen()
        else:
            assert False

        prog, expr,	decls, scales, intvs, cnsts = compiler.visit(ast)

        res = prog, expr
        state = decls, scales, intvs, cnsts, compiler.expTables, compiler.VAR_IDF_INIT

        return res, state
