# %%
from typing import (
    TypeVar,
    cast,
    Generic,
    Union,
    Callable,
    TypeAlias,
    overload,
    TypeGuard,
    Any,
    Optional,
)
import numpy as np

T = TypeVar("T")


def is_same(val1: Any, val2: Any) -> bool:
    try:
        # 型が異なれば False
        if type(val1) != type(val2):
            return False

        # numpy配列の比較
        if isinstance(val1, np.ndarray):
            return np.array_equal(val1, val2)

        # list, tuple の比較
        if isinstance(val1, (list, tuple)):
            if len(val1) != len(val2):
                return False
            # print(all(is_same(v1, v2) for v1, v2 in zip(val1, val2)))
            return all(is_same(v1, v2) for v1, v2 in zip(val1, val2))

        # int, float, str, boolなど基本型は直接比較
        return val1 == val2
    except Exception as e:
        print("is_same function is not support val1 and val2")
        return False  # 比較できないものは同じとみなさない。


# 状態管理クラス。bind()で状態変更時に呼び出したい処理を登録できる。
class State(Generic[T]):
    def __init__(self, value: T):
        self._value = value
        self._observers: list[Callable] = []

    def get(self) -> T:
        return self._value  # 値の参照はここから

    def set_new_value(self, new_value: T) -> None:
        self._value = new_value  # 新しい値をセット
        for observer in self._observers:
            observer()  # 変更時に各observerに通知する

    def force_set(self, new_value: T) -> None:
        self.set_new_value(new_value)

    def set(self, new_value: T) -> None:
        if not is_same(self._value, new_value):
            self.set_new_value(new_value)

    def bind(self, observer: Callable) -> None:
        self._observers.append(observer)  # 変更時に呼び出す為のリストに登録


# 依存しているStateの変更に応じて値が変わるクラス。
class ReactiveState(Generic[T]):
    # formula: State等を用いて最終的にT型の値を返す関数。
    # 例えばlambda: f'value:{state_text.get()}'といった関数を渡す。

    # reliance_states: 依存関係にあるStateをlist形式で羅列する。
    def __init__(self, formula: Callable[[], T], reliance_states: list[State]):
        self.__value = State(formula())  # 通常のStateクラスとは違い、valueがStateである
        self.__formula = formula
        self._observers: list[Callable] = []

        for state in reliance_states:
            # 依存関係にあるStateが変更されたら、再計算処理を実行するようにする
            state.bind(lambda: self.update())

    def get(self) -> T:
        return self.__value.get()

    def force_update(self) -> None:
        # old_value = self.__value.get()
        # # コンストラクタで渡された計算用の関数を再度呼び出し、値を更新する
        # self.__value.set(self.__formula())
        # if old_value != self.__value.get():

        for observer in self._observers:
            observer()  # 変更時に各observerに通知する

    def update(self) -> None:
        old_value = self.__value.get()
        # コンストラクタで渡された計算用の関数を再度呼び出し、値を更新する
        self.__value.set(self.__formula())

        if old_value != self.__value.get():
            for observer in self._observers:
                observer()  # 変更時に各observerに通知する

    def bind(self, observer: Callable) -> None:
        self._observers.append(observer)  # 変更時に呼び出す為のリストに登録


# リアクティブなコンポーネントの引数(ReactiveStateを追加)
StateProperty: TypeAlias = Union[T, State[T], ReactiveState[T]]


# コンポーネント内でpropsに、Stateになる可能性のある引数を渡す。
# StateやReactiveStateが渡された場合、自動でbind_funcを変更検知イベントに登録する
def bind_props(props: list[StateProperty], bind_func: Callable[[], None]) -> None:
    for prop in props:
        if isinstance(prop, State) or isinstance(prop, ReactiveState):
            prop.bind(lambda: bind_func())


# Stateであれば.get()メソッドを呼び出し、通常の変数であればそのまま値を取得する


@overload
def get_prop_value(prop: None) -> None: ...
@overload
def get_prop_value(prop: T | State[T] | ReactiveState[T]) -> T: ...


def get_prop_value(prop):  # type: ignore
    if isinstance(prop, State):
        return prop.get()
    elif isinstance(prop, ReactiveState):
        return prop.get()
    elif prop is None:
        return None
    else:
        return prop


@overload
def is_not_None_state(state: State[T | None]) -> TypeGuard[State[T]]: ...
@overload
def is_not_None_state(
    state: ReactiveState[T | None],
) -> TypeGuard[ReactiveState[T]]: ...


def is_not_None_state(state: State[T | None] | ReactiveState[T | None]):  # type: ignore
    if state.get() is None:
        return False
    else:
        return True


# テスト例
if __name__ == "__main__":
    print(is_same("aaa", "aaa"))  # True
    print(is_same("aaa", "acaa"))  # True
    print(is_same(10, 10))  # True
    print(is_same(10, 11))  # False
    print(is_same([1, 2, 3], [1, 2, 3]))  # True
    print(is_same([1, 2, 3], [1, 2, 4]))  # False
    print(is_same((1, 2), (1, 2)))  # True
    print(is_same([1, 2], (1, 2)))  # False
    arr1 = np.array([1, 2, 3])
    arr2 = np.array([1, 2, 3])
    print(is_same(arr1, arr2))  # True
    arr3 = np.array([1, 2, 4])
    print(is_same(arr1, arr3))  # False
    print(is_same([1, (2, 3), np.array([4, 5])], [1, (2, 3), np.array([4, 5])]))  # True
    print(
        is_same(
            {
                "aaa": np.array([[1, 1, 2], [1, 2, 3]]),
                "bbb": None,
            },
            {
                "aaa": np.array([[1, 1, 2], [1, 2, 3]]),
                "bbb": None,
            },
        )
    )  # not supported
    print(
        is_same(
            {
                "k": {
                    "color": "#991111",
                    "index": [
                        np.int64(0),
                        np.int64(1395),
                        np.int64(1441),
                        np.int64(1464),
                        np.int64(1478),
                        np.int64(1562),
                        np.int64(1835),
                        np.int64(1992),
                    ],
                    "display": False,
                },
                "k2": {
                    "color": "#991111",
                    "index": [
                        np.int64(0),
                        np.int64(1395),
                        np.int64(1441),
                        np.int64(1464),
                        np.int64(1478),
                        np.int64(1562),
                        np.int64(1835),
                        np.int64(1992),
                    ],
                    "display": False,
                },
            },
            {
                "k": {
                    "color": "#991111",
                    "index": [
                        np.int64(0),
                        np.int64(1395),
                        np.int64(1441),
                        np.int64(1464),
                        np.int64(1478),
                        np.int64(1562),
                        np.int64(1835),
                        np.int64(1992),
                    ],
                    "display": False,
                },
                "k2": {
                    "color": "#991111",
                    "index": [
                        np.int64(0),
                        np.int64(1395),
                        np.int64(1441),
                        np.int64(1464),
                        np.int64(1478),
                        np.int64(1562),
                        np.int64(1835),
                        np.int64(1992),
                    ],
                    "display": False,
                },
            },
        )
    )  # True

# %%
