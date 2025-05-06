import math
import operator
from random import randint

import numpy as np
from pyparsing import (
    CaselessKeyword,
    Regex,
    Word,
    alphas,
    alphanums,
    Literal,
    Suppress,
    Forward,
    delimitedList,
    Group,
    ParseException
)

__all__ = ['evaluate_string']

EXPR_STACK = []

BNF = None


class EvaluateStringExpressionError(ParseException):
    def __init__(self, iden: str):
        super().__init__(iden, msg=f"invalid identifier '{iden}'")
        self.iden = iden


def bnf_parser():
    """
    expop   :: '^'
    multop  :: '*' | '/'
    addop   :: '+' | '-'
    integer :: ['+' | '-'] '0'..'9'+
    atom    :: PI | E | real | fn '(' expr ')' | '(' expr ')'
    factor  :: atom [ expop factor ]*
    term    :: factor [ multop factor ]*
    expr    :: term [ addop term ]*
    """

    global BNF

    if BNF is None:
        # use CaselessKeyword for e and pi, to avoid accidentally matching
        # functions that start with 'e' or 'pi' (such as 'exp'); Keyword
        # and CaselessKeyword only match whole words
        e = CaselessKeyword("E")
        pi = CaselessKeyword("PI")

        #
        fnumber = Regex(r"[+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")
        ident = Word(alphas, alphanums + "_$")

        #
        plus, minus, mult, div = map(Literal, "+-*/")
        lpar, rpar = map(Suppress, "()")
        addop = plus | minus
        multop = mult | div
        expop = Literal("^")

        #
        lesser_than = Literal("<")
        greater_than = Literal('>')
        lesser_or_equal = Literal("<=")
        greater_or_equal = Literal(">=")
        compop = lesser_or_equal | greater_or_equal | lesser_than | greater_than

        #
        expr = Forward()
        expr_list = delimitedList(Group(expr))

        # add parse action that replaces the function identifier with a (name, number of args) tuple
        def insert_fn_argcount_tuple(t):
            fn = t.pop(0)
            num_args = len(t[0])
            t.insert(0, (fn, num_args))

        fn_call = (ident + lpar - Group(expr_list) + rpar).setParseAction(
            insert_fn_argcount_tuple
        )
        atom = (
                (addop[...] | compop[...])
                + (
                        (fn_call | pi | e | fnumber | ident).setParseAction(push_first)
                        | Group(lpar + expr + rpar)
                )
        ).setParseAction(push_unary_minus)

        # by defining exponentiation as "atom [ ^ factor ]..." instead of "atom [ ^ atom ]...", we get right-to-left
        # exponents, instead of left-to-right that is, 2^3^2 = 2^(3^2), not (2^3)^2.
        factor = Forward()
        factor <<= atom + (expop + factor).setParseAction(push_first)[...]
        term = factor + (multop + factor).setParseAction(push_first)[...]
        expr <<= term + ((addop | compop) + term).setParseAction(push_first)[...]
        BNF = expr

    return BNF


def push_unary_minus(toks):
    for t in toks:
        if t == "-":
            EXPR_STACK.append("unary -")
        else:
            break


def push_first(toks):
    EXPR_STACK.append(toks[0])


def evaluate_string(s: str, **p) -> str:
    """

    :param s: evaluate string
    :param p:
    :return:
    """
    EXPR_STACK[:] = []

    try:
        bnf_parser().parseString(s, parseAll=True)
    except ParseException as e:
        raise RuntimeError(s) from e

    val = evaluate_stack(EXPR_STACK[:], p)
    return val


epsilon = 1e-12
opn = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
    "<": operator.lt,
    ">": operator.gt,
    "<=": operator.le,
    ">=": operator.ge
}

fn = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "abs": np.abs,
    "trunc": lambda it: int(it),
    "round": np.round,
    "randint": randint,
    "sgn": lambda a: -1 if a < -epsilon else 1 if a > epsilon else 0,
    "multiply": lambda a, b: a * b,
    "hypot": math.hypot,
    'all': np.all
}


def evaluate_stack(s, p: dict[str, float]):
    """

    :param s:
    :param p:
    :return:
    """
    op, num_args = s.pop(), 0
    if isinstance(op, tuple):
        op, num_args = op

    if op == "unary -":
        return -evaluate_stack(s, p)
    elif op in opn:
        # note: operands are pushed onto the stack in reverse order
        op2 = evaluate_stack(s, p)
        op1 = evaluate_stack(s, p)
        return opn[op](op1, op2)
    elif op == "PI":
        return math.pi  # 3.1415926535
    elif op == "E":
        return math.e  # 2.718281828
    elif op in fn:
        # note: args are pushed onto the stack in reverse order
        args = reversed([evaluate_stack(s, p) for _ in range(num_args)])
        if op == 'all':
            return fn[op](list(args))
        else:
            return fn[op](*args)
    elif op in p:
        return p[op]
    elif op[0].isalpha():
        raise EvaluateStringExpressionError(op)
    else:
        # try to evaluate as int first, then as float if int fails
        try:
            return int(op)
        except ValueError:
            return float(op)
