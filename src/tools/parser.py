from typing import Callable, Optional


def to_str(val: float, digit: Optional[int] = None) -> str:
    if digit is not None:
        # 指定桁数でフォーマット
        s = format(val, f".{digit}f")
        # 末尾の余分な0や小数点を削除
        s = s.rstrip("0").rstrip(".")
    else:
        # 自動的に短い表現をしてくれる'g'形式を使用
        s = format(val, "g")
    return s


def parse_float_larger_than_0(s: str) -> float | None:
    if s.endswith("."):
        # raise ValueError(f"Invalid float literal: {s}")
        # do nothing
        return None
    fs = float(s)
    if 0 < fs:
        return fs
    else:
        raise ValueError("value should larger than 0")


def parse_float_0_to_1(s: str) -> float | None:
    # 入力がピリオドで終わっている場合はエラーを発生させる
    if s.endswith("."):
        # raise ValueError(f"Invalid float literal: {s}")
        # do nothing
        return None
    fs = float(s)
    if 0 <= fs <= 1:
        return fs
    else:
        raise ValueError("cx or cy value should 0 to 1")
