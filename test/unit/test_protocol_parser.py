import math
import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal
from pyparsing import ParseException

from stimpyp.math_eval import EXPR_STACK, bnf_parser, evaluate_stack, EvaluateStringExpressionError
from stimpyp.protocol_parser import (
    remove_comments_and_strip,
    lines_to_variables_dict,
    EmptyProtocolError,
    UnassignedVariableError,
    UndefinedVariableError,
    VariableAlreadyDeclaredError,
    eval_assignments,
    eval_dataframe,
    generate_extended_dataframe
)


def parser_test(s, expected, **p) -> None:
    EXPR_STACK[:] = []

    try:
        results = bnf_parser().parseString(s, parseAll=True)
        val = evaluate_stack(EXPR_STACK[:], p)
    except EvaluateStringExpressionError as e:
        print(s, p, "FAILED eval:", str(e), EXPR_STACK)
    except ParseException as pe:
        print(s, p, "FAILED parse:", str(pe))
    else:
        if val == expected:
            print(s, p, "=", val, results, "=>", EXPR_STACK, "SUCCESS")
        else:
            print(s, p, "!!!", val, "!=", expected, results, "=>", EXPR_STACK, "WRONG VALUE")


class TestMathEval:

    def test_eval_parser(self):
        parser_test("t+1", 3, t=2)
        parser_test("9", 9)
        parser_test("-9", -9)
        parser_test("--9", 9)
        parser_test("-E", -math.e)
        parser_test("9 + 3 + 6", 9 + 3 + 6)
        parser_test("9 + 3 / 11", 9 + 3.0 / 11)
        parser_test("(9 + 3)", (9 + 3))
        parser_test("(9+3) / 11", (9 + 3.0) / 11)
        parser_test("9 - 12 - 6", 9 - 12 - 6)
        parser_test("9 - (12 - 6)", 9 - (12 - 6))
        parser_test("2*3.14159", 2 * 3.14159)
        parser_test("3.1415926535*3.1415926535 / 10", 3.1415926535 * 3.1415926535 / 10)
        parser_test("PI * PI / 10", math.pi * math.pi / 10)
        parser_test("PI*PI/10", math.pi * math.pi / 10)
        parser_test("PI^2", math.pi ** 2)
        parser_test("round(PI^2)", round(math.pi ** 2))
        parser_test("6.02E23 * 8.048", 6.02e23 * 8.048)
        parser_test("e / 3", math.e / 3)
        parser_test("sin(PI/2)", math.sin(math.pi / 2))
        parser_test("10+sin(PI/4)^2", 10 + math.sin(math.pi / 4) ** 2)
        parser_test("trunc(E)", int(math.e))
        parser_test("trunc(-E)", int(-math.e))
        parser_test("round(E)", round(math.e))
        parser_test("round(-E)", round(-math.e))
        parser_test("E^PI", math.e ** math.pi)
        parser_test("exp(0)", 1)
        parser_test("exp(1)", math.e)
        parser_test("2^3^2", 2 ** 3 ** 2)
        parser_test("(2^3)^2", (2 ** 3) ** 2)
        parser_test("2^3+2", 2 ** 3 + 2)
        parser_test("2^3+5", 2 ** 3 + 5)
        parser_test("2^9", 2 ** 9)
        parser_test("sgn(-2)", -1)
        parser_test("sgn(0)", 0)
        parser_test("sgn(0.1)", 1)
        parser_test("round(E, 3)", round(math.e, 3))
        parser_test("round(PI^2, 3)", round(math.pi ** 2, 3))
        parser_test("sgn(cos(PI/4))", 1)
        parser_test("sgn(cos(PI/2))", 0)
        parser_test("sgn(cos(PI*3/4))", -1)
        parser_test("+(sgn(cos(PI/4)))", 1)
        parser_test("-(sgn(cos(PI/4)))", -1)
        parser_test("hypot(3, 4)", 5)
        parser_test("multiply(3, 7)", 21)
        parser_test("all(1,1,1)", True)
        # test("all(1,1,1,1,1,0)", False)

        parser_test("0<3", True)
        parser_test("0>3", False)
        parser_test("3>0", True)
        parser_test("3<0", False)
        parser_test("5*(0<3)", 5)
        parser_test("5*(0<3)+2*(0>3)", 5)

        parser_test("3<3", False)
        parser_test("3<=3", True)
        parser_test("0<=3", True)
        parser_test("4<=3", False)
        parser_test("3>=3", True)
        parser_test("3>=0", True)
        parser_test("0>=3", False)

        parser_test("1+1<2*3", True)
        parser_test("1+(1<2)*3", 4)
        parser_test("(1+1<2)*3", 0)

        parser_test("(-60+20*0)*(0<3)+(30*0)*(0>=3)", -60)
        parser_test("(-60+20*4)*(4<3)+(30*4)*(4>=3)", 120)


class TestProtocolParser(unittest.TestCase):
    """Test class for parser"""

    def test_remove_comments_and_strip(self):
        lines = [
            " #4",
            "#2",
            "var_1 = 3 # x",
            "# var_2 = 4 # var_3 = 5 # x",
            "var_2 =",
            "  a b c",
            "  d e f",
        ]
        stripped_prot = remove_comments_and_strip(lines)
        expected_result = [
            "var_1 = 3",
            "var_2 =",
            "a b c",
            "d e f"
        ]
        self.assertListEqual(stripped_prot, expected_result)

    def test_lines_to_variables_dict_success(self):
        assignments = [
            "var_1 = 3",
            "var_2 =",
            "a b c",
            "d>=1 e<=2 f"
        ]  # "d e f"
        variables_dict = lines_to_variables_dict(assignments)
        expected_result = {
            "var_1": "3",
            "var_2": "a b c\nd>=1 e<=2 f"
        }
        self.assertDictEqual(variables_dict, expected_result)

    def test_lines_to_variables_dict_empty_raises_error(self):
        with self.assertRaises(EmptyProtocolError):
            lines_to_variables_dict([])

    def test_lines_to_variables_dict_unassigned_raises_error(self):
        with self.assertRaises(UnassignedVariableError):
            lines_to_variables_dict(["a="])

    def test_lines_to_variables_dict_undefined_variable_raises_error(self):
        with self.assertRaises(UndefinedVariableError):
            lines_to_variables_dict(["=a"])

    def test_lines_to_variables_dict_multiple_declaration_raises_error(self):
        with self.assertRaises(VariableAlreadyDeclaredError):
            lines_to_variables_dict(["a=1", "a=2"])

    def test_eval_assignments(self):
        dummy_assignment_dict = {
            "var1": "1",
            "var2": "2.0",
            "var3": "True",
            "var4": "astring",
            "var5": "[1,2]",
            "var6": "{'v1': 1}",
            "var7": "dur ori other\n1 2 3",
        }
        evaluated_dict = eval_assignments(dummy_assignment_dict, dataframe_var=['var7'])
        self.assertEqual(evaluated_dict["var1"], int(1))
        self.assertEqual(evaluated_dict["var2"], float(2))
        self.assertEqual(evaluated_dict["var3"], True)
        self.assertEqual(evaluated_dict["var4"], "astring")
        self.assertEqual(evaluated_dict["var5"], [1, 2])
        self.assertEqual(evaluated_dict["var6"], {"v1": 1})
        df = pl.DataFrame(data=dict(dur=[1], ori=[2], other=[3]))
        assert_frame_equal(evaluated_dict["var7"], df, check_dtypes=False)

    def test_parse_dataframe(self):
        data_frame_line = """\
n   img   g   dur len   xc  yc   c   width   height   mask
0    0    0   2   3    -15  0   1.0   15      15    circle
1    1    0   2   3     15  0   0.5   15      15    circle
2    0    1   2   3     15  0   1.0   15      15    circle
3-4    1    1   2   3    -15  0   0.5   15      15    circle
"""
        df = eval_dataframe(data_frame_line)
        assert_frame_equal(df, pl.DataFrame(dict(
            n=['0', '1', '2', '3-4'],
            img=[0, 1, 0, 1],
            g=[0, 0, 1, 1],
            dur=[2, 2, 2, 2],
            len=[3, 3, 3, 3],
            xc=[-15, 15, 15, -15],
            yc=[0, 0, 0, 0],
            c=[1.0, 0.5, 1.0, 0.5],
            width=[15, 15, 15, 15],
            height=[15, 15, 15, 15],
            mask=['circle', 'circle', 'circle', 'circle'],
        )))

    def test_generate_extended_dataframe(self):
        df = pl.DataFrame(dict(
            n=['0', '2-4', '1', ],
            v=['1', '{i}+3', '2', ],
            u=[0, 1, 2]
        ))
        df = generate_extended_dataframe(df)

        result = pl.DataFrame({
            'n': [0, 1, 2, 3, 4],
            'v': [1, 2, 5, 6, 7],
            'u': [0, 2, 1, 1, 1]
        })

        print(result)
        assert_frame_equal(df, result)


class TestExtendedProtocol(unittest.TestCase):

    def test_variable_i(self):
        dataframe_string = """\
n      dur     xc   yc   c    sf    tf   ori      width  height pattern
0-11  3       0    0    1    0.04  1    {i}*30   200    200    sqr
12-23  3       0    0    1    0.08  1    {i}*30   200    200    sqr
"""

        df = eval_dataframe(dataframe_string)
        df = generate_extended_dataframe(df)

        n_rows = df.shape[0]
        expected_result = pl.DataFrame(
            {
                'n': pl.Series(np.arange(n_rows)),
                'dur': pl.Series(np.full(n_rows, 3)),
                'xc': pl.Series(np.full(n_rows, 0)),
                'yc': pl.Series(np.full(n_rows, 0)),
                'c': pl.Series(np.full(n_rows, 1)),
                'sf': pl.Series([0.04] * 12 + [0.08] * 12),
                'tf': pl.Series(np.full(n_rows, 1)),
                'ori': pl.Series(np.arange(0, n_rows * 30, 30)),
                'width': pl.Series(np.full(n_rows, 200)),
                'height': pl.Series(np.full(n_rows, 200)),
                'pattern': pl.Series(np.full(n_rows, 'sqr')),
            }
        )

        assert_frame_equal(df, expected_result)


if __name__ == '__main__':
    unittest.main()
