import AST.AST as AST
from AST.ASTVisitor import ASTVisitor

class MtdAST(ASTVisitor):
	def visitInt(self, node:AST.Int, mtd:dict):
		node.metadata.update(mtd)

	def visitFloat(self, node:AST.Float, mtd:dict):
		node.metadata.update(mtd)

	def visitId(self, node:AST.ID, mtd:dict):
		node.metadata.update(mtd)

	def visitDecl(self, node:AST.Decl, mtd:dict):
		node.metadata.update(mtd)

	def visitTransp(self, node:AST.Transp, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitReshape(self, node:AST.Reshape, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitMaxpool(self, node:AST.Maxpool, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)
	
	def visitIndex(self, node:AST.Index, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitUop(self, node:AST.Uop, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitBop1(self, node:AST.Bop1, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr1, mtd)
		self.visit(node.expr2, mtd)

	def visitBop2(self, node:AST.Bop2, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr1, mtd)
		self.visit(node.expr2, mtd)

	def visitFunc(self, node:AST.Func, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitSum(self, node:AST.Sum, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)

	def visitCond(self, node:AST.Cond, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.expr, mtd)
		self.visit(node.trueBlock, mtd)
		self.visit(node.falseBlock, mtd)

	def visitLet(self, node:AST.Let, mtd:dict):
		node.metadata.update(mtd)
		self.visit(node.name, mtd)
		self.visit(node.decl, mtd)
		self.visit(node.expr, mtd)

	def visitUninterpFuncCall(self, node:AST.FuncCall, mtd:dict):
		node.metadata.update(mtd)
		for curArg in node.argsList:
			self.visit(curArg, mtd)
