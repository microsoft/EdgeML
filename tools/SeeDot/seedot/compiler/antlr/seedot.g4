// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

grammar seedot;

expr:	IntConst								# int
	|	FloatConst								# float
	|	Id										# id
	|	'(' intConstList ')'
		In '[' FloatConst ',' FloatConst ']'	# decl
	|	'init' '('
		'[' intConstList ']' ',' FloatConst ')'	# init

	|	expr '^T'								# transp
	|	Reshape '(' expr ','
		'(' intConstList ')' ','
		'(' intConstList ')' ')'				# reshape
	|	expr '[' expr ':+' IntConst ']' 
		('[' expr ':+' IntConst ']')*			# splice
	|	Maxpool '(' expr ',' 
		'{k' IntConst IntConst '}' ','
		'{p' IntConst IntConst IntConst IntConst '}' ','
		'{s' IntConst IntConst '}'  ')'			# maxpool
		
	|	Reverse '(' expr ',' IntConst ')'		# reverse
	|	expr '[' expr ']'						# index
	|	Id '(' expr (',' expr)* ')'				# funcCall

	|	addOp expr								# uop
	|	expr binOp expr							# bop1
	|	expr addOp expr							# bop2
	|	Conv2d '(' expr ',' expr ',' 
		'{s' IntConst IntConst '}' ',' 
		'{p' IntConst IntConst IntConst IntConst'}' ','
		'{d' IntConst IntConst '}' ','
		'{g'IntConst'}' ')' 					# convolution
	|	MbConv '(' expr ',' 
		'[' expr ',' expr ',' expr ']' ','
		'[' expr ',' expr ',' expr ']' ','
		'[' expr ',' expr ',' expr ']' ','
		'{s' IntConst IntConst '}' ',' '{p' IntConst 
		IntConst IntConst IntConst'}' ')' 		# mbconv

	|	specialFunc '(' expr ')'				# func
	|	Sum '(' Id '='
		'[' IntConst ':' IntConst ']' ')' expr  # sum
	|	Loop '(' Id '='
		'[' IntConst ':' IntConst ']'
		',' expr ')' expr						# loop

	|	expr '>=' IntConst '?' expr ':' expr	# cond
	|	Let lhs '=' expr In expr	 			# let
	|	'(' expr ')'							# paren
	;

lhs : 	Id   				 											 # name
	|	Id ('[' expr ':+' IntConst ']') ('[' expr ':+' IntConst ']')*    # leftSplice
	;

addOp	:	ADD
		|	SUB
		;
binOp	:	MUL
		|	SPARSEMUL
		|	MULCIR
		|	ADDCIR
		|	SUBCIR
		;
specialFunc	:	RELU
			|	RELU6
			|	EXP
			|	ARGMAX
			|	SGN
			|	TANH
			|	SIGMOID
			|   NORMALISEL2
			;

ADD		:	'+' ;
SUB		:	'-' ;
MUL		:	'*' ;
SPARSEMUL:	'|*|' ;
MULCIR	:	'<*>' ;
ADDCIR	:	'<+>' ;
SUBCIR	:	'<->' ;

RELU	:	'relu'			;
RELU6	:	'relu6'			;
EXP		:	'exp'    		;
ARGMAX	:	'argmax' 		;
SGN		:	'sgn'    		;
TANH	:	'tanh'   		;
SIGMOID	:	'sigmoid'		;
NORMALISEL2: 'normaliseL2'	;

MbConv	:	'mbconv' ;
Conv2d	:	'conv2d' ;
Reshape	:	'reshape' ;
Maxpool	:	'maxpool' ;
Reverse :   'reverse' ;
Sum		:	'$'       ;
Loop	:	'loop'    ;

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
