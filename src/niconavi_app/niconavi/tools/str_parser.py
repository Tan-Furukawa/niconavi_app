# %%
import sys
from typing import Callable

# ! エラーを返す -> 一秒後に修正される直前の正しい値に修正される。
# ! Noneを返す -> 空欄のままになる


def parse_float(s: str, parse_fn: Callable[[str], float]) -> float | None:
    if s == "":
        return None
    if not "." in s and s != "":
        return int(parse_fn(s))  # 整数のときは、floatではないけど、整数として返す
    else:
        return parse_fn(s)


def parse_larger_than_0(s: str) -> float | None:
    def larger_than_0(s: str) -> float:
        v = float(s)
        if v >= 0:
            return v
        else:
            raise ValueError("")

    return parse_float(s, larger_than_0)

def parse_larger_than_1(s: str) -> float | None:
    def larger_than_1(s: str) -> float:
        v = float(s)
        if v >= 1:
            return v
        else:
            raise ValueError("")

    return parse_float(s, larger_than_1)

def parse_int(s: str) -> int | None:
    if s == "":
        return None
    fs = int(s)
    return fs

def parse_int_smaller_than(max_val: int) -> Callable[[str], int | None]:
    def closure(s:str) -> int | None:
        if s == "":
            return None

        fs = int(s)
        if 0 <= fs and fs <= max_val:
            return fs
        else:
            raise ValueError(
                "invalid value input: percentile value must 0 <= val and val <= 100"
            )
    return closure



def parse_odd_int(s: str) -> int | None:
    if s == "":
        return None
    # 入力がピリオドで終わっている場合はエラーを発生させる
    fs = int(s)
    if fs % 2 == 1:
        return fs
    else:
        raise ValueError("median kernel size must odd number")


def parse_percent(s: str) -> float | None:
    ss = s.split(".")
    if "." in s and ss[-1] == "":
        return None  # 点でおわっているときはなにもしない
    if not "." in s:
        return int(s)  # 整数のときは、floatではないけど、整数として返す
    if s == "":
        return None
    fs = float(s)
    if 0 <= fs and fs <= 100:
        return fs
    else:
        raise ValueError(
            "invalid value input: percentile value must 0 <= val and val <= 100"
        )


if __name__ == "__main__":
    a = parse_larger_than_0("")
    print(a)
    a = parse_int("aaaa")
    print(a)
