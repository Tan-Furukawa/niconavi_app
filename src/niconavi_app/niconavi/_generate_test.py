#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import argparse


def get_functions(file_path: str) -> list[str]:
    """
    指定したPythonファイルからトップレベルの関数名を抽出して返す
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=file_path)
    except SyntaxError:
        return []

    funcs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            funcs.append(node.name)
    return funcs


def mirror_and_generate(src_dir: str, tests_dir: str):
    """
    src_dir 以下のディレクトリ構造を tests_dir に再現し、
    各モジュールの全関数に対してテストスタブを生成する
    """
    src_dir = os.path.abspath(src_dir)
    tests_dir = os.path.abspath(tests_dir)

    for dirpath, dirnames, filenames in os.walk(src_dir):
        # __pycache__ やテスト出力先を除外
        if '__pycache__' in dirpath or dirpath.startswith(tests_dir):
            continue

        rel_dir = os.path.relpath(dirpath, src_dir)
        rel_dir = '' if rel_dir == '.' else rel_dir

        target_dir = os.path.join(tests_dir, rel_dir)
        os.makedirs(target_dir, exist_ok=True)

        # tests パッケージとして __init__.py
        init_py = os.path.join(target_dir, '__init__.py')
        if not os.path.exists(init_py):
            open(init_py, 'w', encoding='utf-8').close()

        for fn in filenames:
            if not fn.endswith('.py') or fn == '__init__.py':
                continue

            src_file = os.path.join(dirpath, fn)
            funcs = get_functions(src_file)
            if not funcs:
                continue

            test_file = os.path.join(target_dir, f'test_{fn}')
            if os.path.exists(test_file):
                # 既にあるなら上書きせずスキップ
                continue

            # テスト雛形を生成
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(f"""
# Auto-generated tests for {os.path.join(rel_dir, fn) or fn}
"""
)
                for func in funcs:
                    stub = (
                        f'def test_{func}():\n'
                        f'    # TODO: {fn} のテストを実装\n'
                        '    assert False, "テストが未実装です"\n'
                        '\n'
                    )
                    f.write(stub)

    print(f'✅ テストスタブ生成完了: `{src_dir}` → `{tests_dir}`')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='既存モジュールの全関数を検出し、テストスタブを生成します'
    )
    parser.add_argument(
        '-s', '--source',
        default='.',
        help='ソースディレクトリ（デフォルト: カレントディレクトリ）'
    )
    parser.add_argument(
        '-d', '--dest',
        default='tests',
        help='テスト出力先ディレクトリ（デフォルト: tests/）'
    )
    args = parser.parse_args()
    mirror_and_generate(args.source, args.dest)
