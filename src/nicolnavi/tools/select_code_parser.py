# %%
from lark import Lark, Transformer, v_args
import operator
from typing import Callable, Any, Dict, Tuple, Optional, cast
from niconavi.tools.type import D1IntArray
from niconavi.type import GrainSelectedResult, Grain
import re
import random


@v_args(inline=True)
class EvalTransformer(Transformer):
    # 演算子に対応するPythonの関数をマッピング
    ops = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    def __init__(self):
        super().__init__()

    def or_(self, left, right):
        return lambda env: left(env) or right(env)

    def and_(self, left, right):
        return lambda env: left(env) and right(env)

    def not_(self, val):
        return lambda env: not val(env)

    def compare_op(self, left, op, right):
        # 例: left > right
        fn = self.ops[op]
        return lambda env: fn(left(env), right(env))

    def add(self, left, right):
        return lambda env: left(env) + right(env)

    def sub(self, left, right):
        return lambda env: left(env) - right(env)

    def mul(self, left, right):
        return lambda env: left(env) * right(env)

    def div(self, left, right):
        return lambda env: left(env) / right(env)

    def var(self, name):
        # 変数参照
        var_name = str(name)
        return lambda env: env[var_name]

    def number(self, token):
        # 数値
        val = float(token)  # intでもいいが、floatにしておく
        return lambda env: val

    def string(self, token):
        # 文字列（"a" のようにクォート付きで入ってくる）
        # Lark の ESCAPED_STRING は両端にクォートがついているので外す
        # 例: "\"a\"" -> "a"
        return lambda env: token.strip('"').strip("'")


def parse_label_and_expr(line: str) -> Tuple[str, Optional[str], str]:
    """
    1行を受け取り、例えば 'x1 [white] : (val1 > 0.5 or val2 <= 10) and val2 == "a"'
    のような文字列を「ラベル」、「色」、および「式」に分割して返す。
    色情報がない場合は color を None とする。
    """
    line = line.strip()
    # 正規表現を使用してラベル、色、式をパース
    pattern = r"^(?P<label>\w+)(?:\s*\[(?P<color>[^\]]*)\])?\s*:\s*(?P<expr>.+)$"
    match = re.match(pattern, line)
    if not match:
        raise ValueError(f"Invalid format: {line}")

    label = match.group("label")
    color = match.group("color")
    expr_str = match.group("expr").strip()

    # 空の色情報や未指定の場合は None とする
    if color is not None:
        color = color.strip()
        if color == "":
            color = None

    return label, color, expr_str


def compile_expression(expr_str: str, parser: Lark) -> Callable[[dict], Any]:
    """
    Lark で expr_str をパースして、EvalTransformer で評価関数(lambda)を作り出す。
    """
    tree = parser.parse(expr_str)
    func = EvalTransformer().transform(tree)
    return func


def parse_user_input(lines, parser: Lark) -> Dict[str, Dict[str, Any]]:
    """
    ユーザーが入力した複数行 (x1 [color]: 式, x2 [color]: 式, ...) をまとめてパースして
    { ラベル: {"color": 色, "func": 評価関数} } の辞書を返す。
    ラベルが重複している場合はエラーを発生させる。
    """
    result = {}
    for line_number, line in enumerate(lines, start=1):
        # 空行はスキップ
        if not line.strip():
            continue

        label, color, expr_str = parse_label_and_expr(line)
        # 重複チェック
        if label in result:
            raise ValueError(f"Duplicate label '{label}' found on line {line_number}.")
        # 式をコンパイル
        func = compile_expression(expr_str, parser)
        result[label] = {"color": color, "func": func}
    return result


def filter_data(
    data_list: list[Grain], compiled_exprs: Dict[str, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    data_list: [{index: 1, val1: 0.1, val2: 'a', val3: 80}, {...}, ...]
    compiled_exprs: { 'x1': {"color": 色, "func": <function>}, 'x2': {...}, ... }
      上記の <function> は「env(dict)を受け取って真偽を返す」形

    戻り値: {'x1': {"color": 色, "index": [マッチするindexの配列]}, 'x2': {"color": 色, "index": [...]}, ...} の形式
    """
    results = {}
    for label, info in compiled_exprs.items():
        results[label] = {"color": info["color"], "index": [], "display": True}

    for i, row in enumerate(data_list):
        for label, info in compiled_exprs.items():
            func = info["func"]
            if func(row):
                results[label]["index"].append(row["index"])
    return results


def split_multiline_input(input_string: str) -> list:
    """
    複数行の文字列を行ごとのリストに分割し、行コメント (// xxx) を削除する関数。
    """
    lines = input_string.splitlines()
    cleaned_lines = []
    for line in lines:
        # '//' より後ろをコメントとして削除
        comment_start = line.find("//")
        if comment_start != -1:
            line = line[:comment_start]
        # 残った部分を strip して空でなければ追加
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    return cleaned_lines


def deduplicate_indices(data: dict) -> dict:
    """
    dataの形式:
    {
      "key名": {
        "color": <str>,
        "index": <List[int]>,
        "display": <bool> (他にあってもOK)
      },
      ...
    }

    1. 同一キー内で重複するインデックスは1つだけ残す。
    2. 異なるキー間で重複するインデックスがあれば、最後に登場したキーだけに残し、
       それ以前のキーからは削除する。
    """
    # indexがどのkeyに所属しているかを記憶する辞書
    # 例: { 253: "grt", 81: "quartz", ... }

    index_to_key = {}

    # 実際に加工して返す用の新しい辞書を作る
    # ここでは元の構造をベースに、"index" だけ書き換える方法を取る
    result = {}
    for key in data:
        # このkeyに対応するデータをコピー（浅いコピー）しておく
        # ※deepcopyが必要になるケースもありますが、ここでは index 以外は
        #   そのままで良い想定としています。
        result[key] = dict(data[key])

        new_index_list = []
        seen_in_this_key = set()  # このkey内で既に登録したindex

        # 元のindexリストを順に走査
        for idx in data[key]["index"]:
            # 同一キー内でまだ出現していないか？
            if idx not in seen_in_this_key:
                # もし既に他のキーに所属していたら、そのキーからは削除する
                if idx in index_to_key:
                    old_key = index_to_key[idx]
                    if old_key != key:
                        # 以前登録されていたキーから idx を除去
                        # 既にresultに入っているはずなので、そこから取り除く
                        old_list = result[old_key]["index"]
                        # 存在すれば削除 (同一キーリスト内から最後に移動するイメージ)
                        if idx in old_list:
                            old_list.remove(idx)

                # このキーに登録する
                new_index_list.append(idx)
                seen_in_this_key.add(idx)
                index_to_key[idx] = key

        # 重複を除去して整理したインデックスリストを格納
        result[key]["index"] = new_index_list

    return result


def make_select_grain_fn_by_str_function() -> (
    Callable[[list[Grain], str], Dict[str, GrainSelectedResult]]
):
    logic_grammar = r"""
        ?start: expr

        ?expr: or_expr

        ?or_expr: and_expr
                | or_expr "or" and_expr   -> or_
                | or_expr "OR" and_expr   -> or_
        ?and_expr: not_expr
                | and_expr "and" not_expr -> and_
                | and_expr "AND" not_expr -> and_
        ?not_expr: compare
                | "not" not_expr          -> not_
                | "NOT" not_expr          -> not_
        ?compare: sum_expr COMP_OP sum_expr    -> compare_op
                | sum_expr
        ?sum_expr: product
                | sum_expr "+" product     -> add
                | sum_expr "-" product     -> sub
        ?product: atom
                | product "*" atom         -> mul
                | product "/" atom         -> div
        ?atom: NAME                         -> var
            | NUMBER                       -> number
            | ESCAPED_STRING               -> string
            | "(" expr ")"

        COMP_OP: ">" | ">=" | "<" | "<=" | "==" | "!="
        NAME: /[a-zA-Z_]\w*/

        %import common.NUMBER
        %import common.ESCAPED_STRING
        %import common.WS
        %ignore WS
    """

    parser = Lark(logic_grammar, start="start", parser="lalr")

    def filter_by_str(
        data: list[Grain], user_input_str: str
    ) -> Dict[str, GrainSelectedResult]:
        # (1) まず行ごとのリストを作り、コメントを除去
        user_input_lines = split_multiline_input(user_input_str)

        # (2) 入力をパースして {ラベル: {"color": 色, "func": 評価関数} } の辞書を作る
        compiled_exprs = parse_user_input(user_input_lines, parser)

        # (3) データに対して評価を行い、indexリストを収集
        results = filter_data(data, compiled_exprs)
        results = deduplicate_indices(results)
        return cast(Dict[str, GrainSelectedResult], results)

    return filter_by_str

def generate_random_color():
    """
    ランダムな16進数カラーコード（例：#A1B2C3）を生成して返す
    """
    return "#{:06X}".format(random.randint(0, 0xFFFFFF))

def add_random_colors_to_input(user_input):
    """
    ユーザ入力の各行を処理し、色指定（角括弧内）が空または存在しない場合に、
    ランダムな16進数カラーコードを生成して挿入する関数

    Parameters:
        user_input (str): 複数行の入力文字列

    Returns:
        str: 色指定が補完された新たな文字列
    """
    output_lines = []
    for line in user_input.splitlines():
        # コメント行または空行はそのまま出力
        if line.strip().startswith("//") or line.strip() == "":
            output_lines.append(line)
            continue

        # ":" を含まない行はそのまま出力
        if ":" not in line:
            output_lines.append(line)
            continue

        # ":" の前後で文字列を分割
        colon_index = line.find(":")
        before_colon = line[:colon_index]
        after_colon = line[colon_index:]
        
        # 角括弧[...] の存在を確認
        bracket_match = re.search(r"\[(.*?)\]", before_colon)
        if bracket_match:
            # 角括弧内が空の場合、ランダムカラーを生成して挿入する
            if bracket_match.group(1).strip() == "":
                random_color = generate_random_color()
                new_bracket = f"[{random_color}]"
                new_before = re.sub(r"\[\s*\]", new_bracket, before_colon)
                new_line = new_before + after_colon
                output_lines.append(new_line)
            else:
                # すでに色指定がある場合は変更せずそのまま出力
                output_lines.append(line)
        else:
            # 角括弧自体が無い場合、コロン直前にランダムカラーを追加
            random_color = generate_random_color()
            new_before = before_colon.rstrip() + f" [{random_color}]"
            new_line = new_before + after_colon
            output_lines.append(new_line)
    return "\n".join(output_lines)

# サンプルのユーザ入力
user_input = """
    // this is comment
    x1 [white] : (val1 > 0.5 or val3 <= 10) and val2 == "a" // comment x1
    x2 []: (val1 == 0.1 and val2 != "b")
    x3 [#7FFFD4]: (val1 == 0.8 and val2 != "b") // comment
    x4: (val1 == 0.3 and val2 != "c")
    // another comment
    // comment
"""


# 実際に使用する関数
select_grain = make_select_grain_fn_by_str_function()

if __name__ == "__main__":

    # サンプルデータ
    data = [
        {"index": 1, "val1": 0.1, "val2": "a", "val3": 80},
        {"index": 2, "val1": 0.7, "val2": "a", "val3": 5},
        {"index": 3, "val1": 0.6, "val2": "b", "val3": 10},
        {"index": 4, "val1": 0.8, "val2": "z", "val3": 9},
        {"index": 5, "val1": 1.0, "val2": "a", "val3": 2},
        {"index": 6, "val1": 0.3, "val2": "c", "val3": 15},
    ]

    # ユーザー入力の例（複数行 + 行コメント）
    user_input = """
        // this is comment
        x1 [white] : (val1 > 0.5 or val3 <= 10) and val2 == "a" // comment x1
        x2 []: (val1 == 0.1 and val2 != "b")
        x3 [#7FFFD4]: (val1 == 0.8 and val2 != "b") // comment
        x4: (val1 == 0.3 and val2 != "c")
        // another comment
        // comment
    """

#     user_input = """
#  other: R >=0
#     quartz: R > 40 and size > 200
#     mica: R > 500
#         """

    print(add_random_colors_to_input(user_input))
    # result = select_grain(data, user_input)
    # print(result)
    # 期待される出力:
    # {
    #   'x1': {"color": "white", "index": [2, 5]},
    #   'x2': {"color": None, "index": [1]},
    #   'x3': {"color": "#7FFFD4", "index": [4]},
    #   'x4': {"color": None, "index": []}
    # }
