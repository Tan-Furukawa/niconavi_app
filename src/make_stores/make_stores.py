# %%
import ast
from typing import Any


# --- ユーティリティ関数 ---
def generate_computation_result_state_class(
    class_def: ast.ClassDef, special_fields: list[tuple[str, str]]
) -> str:
    """
    Code A の ComputationResult クラス定義から、対応する
    ComputationResultState クラスのコードを生成する専用関数です。
    special_fields に該当するプロパティは、<OriginalClassName>State を用いて初期化します。

    例:
        special_fields = [
            ("grain_detection_parameters", "GrainDetectionParameters"),
            ("plot_parameters", "PlotParameters"),
            ("optical_parameters", "OpticalParameters"),
            ("tilt_image_info", "TiltImageInfo"),
            ("color_chart", "ColorChart"),
        ]

        該当フィールドの場合:
            self.grain_detection_parameters: GrainDetectionParametersState = GrainDetectionParametersState()
    """
    orig_class_name = class_def.name
    state_class_name = orig_class_name + "State"

    # __init__ メソッドを探索
    init_func = None
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_func = node
            break
    if init_func is None:
        return ""

    # special_fields を辞書に変換（フィールド名 -> オリジナルクラス名）
    special_dict = {field: orig for field, orig in special_fields}

    # __init__ の引数（self を除く）
    args = init_func.args.args[1:]  # 最初の self は除外
    num_args = len(args)
    num_defaults = len(init_func.args.defaults)
    # 既定値は引数リストの末尾に対応するので、前半は None で埋める
    defaults = [None] * (num_args - num_defaults) + init_func.args.defaults

    lines = []
    for arg, default in zip(args, defaults):
        param_name = arg.arg
        if param_name in special_dict:
            # 該当フィールドの場合は、<OriginalClassName>State を使用
            special_state_type = special_dict[param_name] + "State"
            line = f"        self.{param_name}: {special_state_type} = {special_state_type}()"
        else:
            # 通常の State クラス生成
            if arg.annotation:
                param_type = ast.unparse(arg.annotation)
            else:
                param_type = "Any"
            if default is not None:
                default_value = ast.unparse(default)
            else:
                default_value = "None"
            line = f"        self.{param_name}: State[{param_type}] = State({default_value})"
        lines.append(line)

    init_body = "\n".join(lines)
    class_code = (
        f"class {state_class_name}:\n"
        f"    def __init__(self) -> None:\n"
        f"{init_body}\n"
    )
    return class_code


def generate_state_class(class_def: ast.ClassDef) -> str:
    """
    Code A のクラス定義から、対応する State クラスのコードを生成する
    （各引数は「self.<name>: State[<型>] = State(<既定値>)」とする）
    """
    orig_class_name = class_def.name
    state_class_name = orig_class_name + "State"

    # __init__ メソッドを探索
    init_func = None
    for node in class_def.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_func = node
            break
    if init_func is None:
        return ""

    # __init__ の引数（self を除く）
    args = init_func.args.args[1:]  # 最初の self は除外
    num_args = len(args)
    num_defaults = len(init_func.args.defaults)
    # 既定値は引数リストの末尾に対応するので、前半は None で埋める
    defaults = [None] * (num_args - num_defaults) + init_func.args.defaults

    lines = []
    for arg, default in zip(args, defaults):
        param_name = arg.arg
        # 型注釈の文字列化（無ければ Any とする）
        if arg.annotation:
            param_type = ast.unparse(arg.annotation)
        else:
            param_type = "Any"
        # 既定値の文字列化（None もそのまま）
        if default is not None:
            default_value = ast.unparse(default)
        else:
            default_value = "None"
        # 出力行の生成
        line = (
            f"        self.{param_name}: State[{param_type}] = State({default_value})"
        )
        lines.append(line)

    init_body = "\n".join(lines)
    class_code = (
        f"class {state_class_name}:\n"
        f"    def __init__(self) -> None:\n"
        f"{init_body}\n"
    )
    return class_code


def generate_conversion_function(class_def: ast.ClassDef) -> str:
    """
    Stateクラスから元のクラスへ変換する関数 as_クラス名 を生成する
    """
    orig_class_name = class_def.name
    state_class_name = orig_class_name + "State"
    if orig_class_name == "ComputationResult":
        return "\n"
    func_name = f"as_{orig_class_name}"
    lines = []
    lines.append(f"def {func_name}(param: {state_class_name}) -> {orig_class_name}:")
    lines.append("    res_dict = {}")
    lines.append("    for key, value in param.__dict__.items():")
    lines.append("        if isinstance(value, State):")
    lines.append("            res_dict[key] = value.get()")
    lines.append(f"    return {orig_class_name}(**res_dict)")
    return "\n".join(lines)


def generate_save_function_for_field(field_name: str, orig_class_name: str) -> str:
    """
    指定フィールド（例: grain_detection_parameters）に対応する
    save_in_～State 関数を生成する。
    関数名は save_in_<orig_class_name>State とし、
    stores.computation_result.<field_name>.__dict__ に対して State オブジェクトの set() を呼ぶ。
    """
    func_name = f"save_in_{orig_class_name}State"
    lines = []
    lines.append(f"def {func_name}(param: {orig_class_name}, stores: Stores) -> None:")
    lines.append(f"    p_dict = param.__dict__")
    lines.append(f"    d = stores.computation_result.{field_name}.__dict__")
    lines.append("    for key in d:")
    lines.append("        if isinstance(d[key], State):")
    lines.append("            d[key].set(p_dict[key])")
    return "\n".join(lines)


def generate_save_in_computation_result_state(
    special_fields: list[tuple[str, str]],
) -> str:
    """
    ComputationResult 用の特別な save_in_ComputationResultState 関数を生成する。
    special_fields は (フィールド名, 元のクラス名) のリストで、
    例: [("grain_detection_parameters", "GrainDetectionParameters"), ...]
    """
    lines = []
    lines.append(
        "def save_in_ComputationResultState(param: ComputationResult, stores: Stores) -> None:"
    )
    lines.append("    p_dict = param.__dict__")
    lines.append("    d = stores.computation_result.__dict__")
    lines.append("    for key in d:")
    lines.append("        if isinstance(d[key], State):")
    lines.append("            d[key].set(p_dict[key])")
    # special_fields に対応する分岐を追加
    for field_name, orig_class_name in special_fields:
        state_class_name = orig_class_name + "State"
        func_name = f"save_in_{orig_class_name}State"
        lines.append(f"        elif isinstance(d[key], {state_class_name}):")
        lines.append(f"            {func_name}(p_dict[key], stores)")
    lines.append("        else:")
    lines.append(
        '            raise ValueError("unexpected type occurred in ComputationResult")'
    )
    return "\n".join(lines)


def generate_as_ComputationResult(special_fields: list[tuple[str, str]]) -> str:
    lines = []
    lines.append(
        "def as_ComputationResult(param: ComputationResultState) -> ComputationResult:"
    )
    lines.append("    res_dict = {}")
    lines.append("    for key in param.__dict__:")
    lines.append("        if isinstance(param.__dict__[key], State):")
    lines.append("            res_dict[key] = param.__dict__[key].get()")
    for field_name, orig_class_name in special_fields:
        state_class_name = orig_class_name + "State"
        lines.append(
            f"        elif isinstance(param.__dict__[key], {state_class_name}):"
        )
        lines.append(
            f"            res_dict[key] = as_{orig_class_name}(param.__dict__[key])"
        )
    lines.append("        else:")
    lines.append(
        '            raise ValueError("unexpected type occurred in ComputationResultState")'
    )
    lines.append("")
    lines.append("    return ComputationResult(**res_dict)")
    return "\n".join(lines)


# --- 全体の生成処理 ---


def generate_code_b(source_code: str, template_code: str) -> str:
    """
    Code A のソースコードから、State クラス、変換関数、各 save_in_～ 関数、
    および ComputationResult 用の特別な保存関数を生成し、
    template_code 内の "# {{{from make_stores.py}}}" の下に挿入して返します。
    """
    import ast

    #! -----------------------------------------------------------------
    #! ComputationResultState中でStateでないオブジェクトの一覧
    #! -----------------------------------------------------------------
    special_fields = [
        ("grain_detection_parameters", "GrainDetectionParameters"),
        ("plot_parameters", "PlotParameters"),
        ("optical_parameters", "OpticalParameters"),
        ("tilt_image_info", "TiltImageInfo"),
        ("color_chart", "ColorChart"),
    ]
    #! -----------------------------------------------------------------

    tree = ast.parse(source_code)
    output_lines = []

    # Stateクラスと変換関数を生成
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            # ComputationResult 以外も通常通り生成
            if node.name == "ComputationResult":
                state_class_code = generate_computation_result_state_class(
                    node, special_fields
                )
            else:
                state_class_code = generate_state_class(node)
            if state_class_code:
                output_lines.append(state_class_code)
                conversion_code = generate_conversion_function(node)
                output_lines.append("\n" + conversion_code + "\n")

    # ※ 以下、特に ComputationResult 内の各コンポーネントの保存関数を生成
    for field_name, orig_class_name in special_fields:
        save_func = generate_save_function_for_field(field_name, orig_class_name)
        output_lines.append(save_func + "\n")

    # ComputationResult 用の保存関数と変換関数を特別扱いで生成
    save_comp_result_func = generate_save_in_computation_result_state(special_fields)
    as_comp_result_func = generate_as_ComputationResult(special_fields)
    output_lines.append(save_comp_result_func + "\n")
    output_lines.append(as_comp_result_func + "\n")

    generated_code = "\n".join(output_lines)

    # テンプレート内の "# {{{from make_stores.py}}}" の下に生成コードを挿入
    marker = "# {{{from make_stores.py}}}"
    if marker in template_code:
        parts = template_code.split(marker)
        new_template = parts[0] + marker + "\n" + generated_code + "\n" + parts[1]
    else:
        new_template = template_code + "\n" + generated_code

    return new_template


def main() -> None:
    # Code A が記述されたファイル（例: code_a.py）を読み込む

    with open("template.py", "r", encoding="utf-8") as f:
        template_code = f.read()
    with open("../../../niconavi/type.py", "r", encoding="utf-8") as f:
        source_code = f.read()

    auto_msg = "\n#!------------------------------------------------------\n#! This code is automatically generated.\n#! このコードは自動生成されています。手動で編集をしないでください。\n#!------------------------------------------------------\n\n"

    generated_code = generate_code_b(source_code, template_code)

    # 生成された Code B をファイルに出力する（例: code_b_generated.py）
    with open("../stores.py", "w", encoding="utf-8") as f:
        f.write(auto_msg + generated_code)
    print(
        "State クラス、変換関数、save_in_～ 関数群を code_b_generated.py に出力しました。"
    )


if __name__ == "__main__":
    main()

# %%
