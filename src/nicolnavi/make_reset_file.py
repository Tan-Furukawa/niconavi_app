#%%
import ast
import json
from pprint import pprint

INPUT_FILE = "type.py"


def extract_keys_from_dict(node: ast.Dict) -> dict:
    """
    辞書リテラルの AST ノードから、
    キー（文字列）のみを抽出します。
    値がさらに dict リテラルや
    TiltImageInfo( **{ ... } ) の形の場合は再帰的に抽出します。
    """
    result = {}
    for key_node, value_node in zip(node.keys, node.values):
        # 辞書展開（**xxx）の場合は key_node が None になっているのでスキップ
        if key_node is None:
            continue
        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
            key_name = key_node.value
            if isinstance(value_node, ast.Dict):
                # 値が辞書リテラルの場合、再帰的にキーを抽出
                result[key_name] = extract_keys_from_dict(value_node)
            elif isinstance(value_node, ast.Call):
                # TiltImageInfo(...) のような場合、内部の辞書リテラルを探す
                if isinstance(value_node.func, ast.Name) and value_node.func.id == "TiltImageInfo":
                    # 通常、TiltImageInfo は **{ ... } の形で渡されるため、そのキーワードを探す
                    for kw in value_node.keywords:
                        # キーワード引数で arg が None は **展開部分
                        if kw.arg is None and isinstance(kw.value, ast.Dict):
                            result[key_name] = extract_keys_from_dict(kw.value)
                            break
                    else:
                        result[key_name] = None
                else:
                    result[key_name] = None
            else:
                result[key_name] = None
    return result


def merge_key_dict(d1: dict, d2: dict) -> dict:
    """
    2 つのキー辞書（再帰的なツリー構造）をマージします。
    同じキーがあれば、両方が dict なら再帰的にマージ、
    そうでなければ None として扱います。
    """
    merged = dict(d1)
    for k, v in d2.items():
        if k in merged:
            if isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = merge_key_dict(merged[k], v)
            else:
                merged[k] = None
        else:
            merged[k] = v
    return merged


def extract_reset_keys_from_return(ret_node: ast.Return) -> dict:
    """
    Return 節のノードから、もし値が ComputationResult(**{ ... }) となっていれば
    辞書リテラル内のキーを抽出します。
    """
    keys = {}
    if isinstance(ret_node.value, ast.Call):
        call_node = ret_node.value
        # ComputationResult かどうかのチェック
        if isinstance(call_node.func, ast.Name) and call_node.func.id == "ComputationResult":
            # 引数として渡される **{ ... } 部分は keywords リストのうち arg==None のもの
            for kw in call_node.keywords:
                if kw.arg is None and isinstance(kw.value, ast.Dict):
                    keys = merge_key_dict(keys, extract_keys_from_dict(kw.value))
    return keys


def extract_reset_keys_from_function(fn_node: ast.FunctionDef) -> dict:
    """
    関数定義内のすべての Return 節から、リセットすべきキーの union を抽出する。
    """
    reset_keys = {}
    for node in ast.walk(fn_node):
        if isinstance(node, ast.Return):
            keys = extract_reset_keys_from_return(node)
            reset_keys = merge_key_dict(reset_keys, keys)
    return reset_keys


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)

    # 関数ごとにリセットすべきキーを抽出する
    functions_reset_keys = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            keys = extract_reset_keys_from_function(node)
            functions_reset_keys[node.name] = keys

    print("抽出されたリセットキー（関数ごと）：")
    pprint(functions_reset_keys)
    # JSON形式で出力したい場合は以下も利用可能
    # print(json.dumps(functions_reset_keys, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
