# %%
from lark import Lark, Transformer, Token
from typing import Callable, Dict, Any, List

grammar = r"""
?start: expr

?expr: or_expr

?or_expr: and_expr
        | or_expr ("or"|"OR") and_expr   -> or_

?and_expr: not_expr
         | and_expr ("and"|"AND") not_expr -> and_

?not_expr: compare
         | ("not"|"NOT") not_expr          -> not_

?compare: sum_expr (COMP_OP sum_expr)?      -> compare_op

?sum_expr: product
         | sum_expr "+" product            -> add
         | sum_expr "-" product            -> sub

?product: atom
        | product "*" atom                 -> mul
        | product "/" atom                 -> div

//---------------------------
// atom: 変数 or 関数呼び出し or 数値 or 括弧
//---------------------------
?atom: function_call
     | VARIABLE
     | NUMBER
     | "(" expr ")"

//---------------------------
// 関数呼び出し
//---------------------------
function_call: FUNCTION_NAME "(" arguments? ")"

arguments: expr ("," expr)*

//---------------------------
// 演算子など
//---------------------------
COMP_OP: ">" | ">=" | "<" | "<=" | "==" | "!="

//---------------------------
// トークンの定義
//---------------------------
FUNCTION_NAME: /[a-zA-Z_]\w*/   // 例: dist, squared, myFn
VARIABLE: /[a-zA-Z_]\w*_\d+/    // 例: a_1, b_2, var_10
NUMBER: /[0-9]+(\.[0-9]+)?/

%import common.WS
%ignore WS
"""


class ExpressionTransformer(Transformer):
    def __init__(self, fn_map: Dict[str, Callable]):
        super().__init__()
        self.fn_map = fn_map  # {"dist": lambda arg1, arg2: ..., ...}

    # 論理演算子
    def or_(self, items):
        left, right = items
        return f"({left}) or ({right})"

    def and_(self, items):
        left, right = items
        return f"({left}) and ({right})"

    def not_(self, items):
        (val,) = items
        return f"not ({val})"

    # 比較演算子
    def compare_op(self, items):
        if len(items) == 1:
            return items[0]
        left, op, right = items
        return f"({left}) {op} ({right})"

    # 四則演算
    def add(self, items):
        left, right = items
        return f"({left}) + ({right})"

    def sub(self, items):
        left, right = items
        return f"({left}) - ({right})"

    def mul(self, items):
        left, right = items
        return f"({left}) * ({right})"

    def div(self, items):
        left, right = items
        return f"({left}) / ({right})"

    # 変数 (a_1, b_2 など)
    def VARIABLE(self, token: Token):
        t = str(token)
        name, idx = t.rsplit("_", 1)
        if idx == "1":
            return f"x['{name}']"
        elif idx == "2":
            return f"y['{name}']"
        else:
            raise ValueError(f"Invalid variable index: {t}")

    # 数値リテラル
    def NUMBER(self, token: Token):
        return str(token)

    # 関数呼び出し
    def function_call(self, items):
        """
        function_call: FUNCTION_NAME "(" arguments? ")"
        items[0] = 関数名 (Token)
        items[1] = 引数リスト (list[str]) or None
        """
        fn_name = str(items[0])
        args_exprs = []
        if len(items) > 1 and items[1] is not None:
            # arguments がある場合
            args_exprs = items[1] if isinstance(items[1], list) else [items[1]]

        if fn_name not in self.fn_map:
            raise ValueError(f"Undefined function: {fn_name}")

        return self._apply_predefined_fn(fn_name, args_exprs)

    def _apply_predefined_fn(self, fn_name: str, args: List[str]) -> str:
        fn = self.fn_map[fn_name]
        return fn(*args)

    def arguments(self, items):
        # expr ("," expr)* → [expr1, expr2, ...]
        return items


def build_function(
    code: str, predefined_fn_list: List[Dict[str, Any]]
) -> Callable[[Dict[str, Any], Dict[str, Any]], bool]:
    # 事前定義関数を "fn_name -> lambda" の形にまとめる
    fn_map = {}
    for fn_def in predefined_fn_list:
        name = fn_def["fn_name"]
        fn_map[name] = fn_def["exec"]

    parser = Lark(grammar, parser="lalr", start="start")
    tree = parser.parse(code)
    transformer = ExpressionTransformer(fn_map=fn_map)
    expr_str = transformer.transform(tree)

    return eval(f"lambda x, y: {expr_str}")


# ------------ テスト ------------
if __name__ == "__main__":
    from niconavi.type import ComputationResult
    import matplotlib.pyplot as plt
    import pandas as pd
    # テスト1: 事前定義関数なし
    code_example1 = "a_1 < a_2 and b_1 - b_2 > 100"
    fn_list1 = []
    fn1 = build_function(code_example1, fn_list1)
    x_data = {"a": 1.5, "b": 300.0}
    y_data = {"a": 2.0, "b": 5.0}
    print("\n--- Example 1 ---")
    print("Code:", code_example1)
    print("Result:", fn1(x_data, y_data))  # => True

    # テスト2: 関数 dist, squared を使う
    # dist(a_1, a_2) => abs(x['a'] - y['a'])
    # squared(b_2)  => (y['b']**2)
    fn_list2 = [
        {"fn_name": "dist", "exec": lambda arg1, arg2: f"abs({arg1} - {arg2})"},
        {
            "fn_name": "dist90",
            "exec": lambda arg1, arg2: f"min(abs({arg1} - {arg2}), 90 - abs({arg1} - {arg2}))",
        },
        {
            "fn_name": "dist180",
            "exec": lambda arg1, arg2: f"min(abs({arg1} - {arg2}), 180 - abs({arg1} - {arg2}))",
        },
        {"fn_name": "squared50", "exec": lambda arg1: f"({arg1}**2 + 50)"},
    ]

    # code_example2 = "dist(a_1, a_2) < 10 and a_1 > 10 or squared(b_2) > 100"
    code_example2 = "dist90(c_1, c_2) < 1"

    fn2 = build_function(code_example2, fn_list2)

    x_data2 = {"a": 9.0, "b": 3.0, "c": 1}
    y_data2 = {"a": 5.0, "b": 5.0, "c": 90}

    print("\n--- Example 2 ---")
    print("Code:", code_example2)
    print("Result:", fn2(x_data2, y_data2))  # => True

    # # デバッグ: 生成された式を見たい場合
    # parser = Lark(grammar, parser="lalr", start="start")
    # tree = parser.parse(code_example2)
    # expr_str_debug = ExpressionTransformer(
    #     fn_map={
    #         "dist": lambda a1, a2: f"abs({a1} - {a2})",
    #         "squared": lambda a1: f"({a1}**2 + 50)",
    #     }
    # ).transform(tree)
    # print("Generated Expression:", expr_str_debug)

    print("\n--- Example 3 ---")
    r: ComputationResult = pd.read_pickle(
        "../../test/data/output/yamagami_cross_before_grain_classification.pkl"
    )

    fn3 = build_function("dist(R_1, R_2) < 10", fn_list2)
    l = r.grain_list
    print(l[0])
    print(fn3(l[0], l[1]))
    # サンプル1: "val1" の差が 1.0 未満であればマージする例