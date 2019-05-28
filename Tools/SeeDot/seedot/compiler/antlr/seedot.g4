// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

grammar seedot;

expr:	IntConst								# int
	|	FloatConst								# float
	|	Id										# id
	|	'(' intConstList ')'
		In '[' FloatConst ',' FloatConst ']'	# decl

	|	expr '^T'								# transp
	|	Reshape '(' expr ','
		'(' intConstList ')' ','
		'(' intConstList ')' ')'				# reshape
	|	Maxpool '(' expr ',' IntConst ')'		# maxpool
	|	expr '[' expr ']'						# index
	|	Id '(' expr (',' expr)* ')'				# funcCall

	|	addOp expr								# uop
	|	expr binOp expr							# bop1
	|	expr addOp expr							# bop2

	|	specialFunc '(' expr ')'				# func
	|	Sum '(' Id '='
		'[' IntConst ':' IntConst ']' ')' expr  # sum

	|	expr '>=' IntConst '?' expr ':' expr	# cond
	|	Let Id '=' expr In expr					# let
	|	'(' expr ')'							# paren
	;

addOp	:	ADD
		|	SUB
		;
binOp	:	MUL
		|	SPARSEMUL
		|	MULCIR
		|	CONV
		|	ADDCIR
		|	SUBCIR
		;
specialFunc	:	RELU
			|	EXP
			|	ARGMAX
			|	SGN
			|	TANH
			;

ADD		:	'+' ;
SUB		:	'-' ;
MUL		:	'*' ;
SPARSEMUL:	'|*|' ;
MULCIR	:	'<*>' ;
CONV	:	'#' ;
ADDCIR	:	'<+>' ;
SUBCIR	:	'<->' ;

RELU	:	'relu'   ;
EXP		:	'exp'    ;
ARGMAX	:	'argmax' ;
SGN		:	'sgn'    ;
TANH	:	'tanh'   ;

Reshape	:	'reshape' ;
Maxpool	:	'maxpool' ;
Sum		:	'$'       ;

Let		:	'let' ;
In		:	'in'  ; 


Id	:	Nondigit (Nondigit | Digit | '\'')* ;
fragment Nondigit	:	[a-zA-Z_] ;

intConstList	:	IntConst (',' IntConst)* ;
IntConst	:	Digit+ ;
fragment Digit	:	[0-9] ;

FloatConst	:	Sign? FracConst ExpntPart?
			|	Sign? Digit+    ExpntPart
			;
fragment FracConst	:	(Digit+)? '.' (Digit+)
					|	(Digit+)  '.'
					;
fragment ExpntPart	:	[eE] Sign? (Digit+) ;
fragment Sign		:	[+-] ;


WS	:	[ \t\r\n]+ -> skip ;	// skip spaces, tabs, newlines
LineComment	:	'//' ~[\r\n]* -> channel(HIDDEN) ;
