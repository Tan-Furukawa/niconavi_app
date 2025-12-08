# %%
from components.log_view import update_logs
from components.progress_bar import update_progress_bar
from logging import Logger
from stores import (
    Stores,
    as_ComputationResult,
    save_in_ComputationResultState,
)
import traceback


# | 番号帯     | 大分類                   | 想定されるサブカテゴリや例                                                             |
# |:-----------|:------------------------|:--------------------------------------------------------------------------------------|
# | **1000台** | **入力・バリデーション** |  - 入力が空・不正フォーマット<br> - 必須パラメータ不足<br> - 桁数・範囲のバリデーション |
# | **2000台** | **認証・認可**           |  - ログイン失敗<br> - セッション切れ<br> - アクセス権限不足                           |
# | **3000台** | **DB / ファイルI/O**     |  - DB接続失敗<br> - トランザクションロールバック<br> - ファイルの読み書き失敗           |
# | **4000台** | **ネットワーク通信**     |  - タイムアウト<br> - DNS解決失敗<br> - 接続拒否、HTTPSエラー                          |
# | **5000台** | **外部サービス連携**     |  - 外部APIエラー<br> - OAuthトークン関連<br> - サードパーティサービス障害               |
# | **6000台** | **ビジネスロジック**     |  - 在庫不足<br> - 業務規則違反<br> - ユースケース要件のバリデーション                  |
# | **7000台** | **リソース管理**         |  - メモリ不足<br> - ファイルロック衝突<br> - スレッドプール不足                        |
# | **8000台** | **致命的でない計算エラー**  |出ることが想定される計算エラー / 出ても計算が破綻しないエラー|
# | **9000台** | **想定外・システム障害** |  - 致命的エラー<br> - 内部で例外が投げられたがハンドル不可<br> - 何らかの不明障害        |

ERROR_MESSAGE = {
    1002: "Please select the XPL video before starting.",
    1003: "No video was selected",
    1004: "Unable to recalculate the stage center. To adjust it, restart from the beginning.",
    8001: "Failed to locate the center. Please determine the rotation center manually.",
    8002: "The XPL movie appears to use the wrong rotation direction. Clockwise stage rotation is not allowed. Re-shoot the movies with counter-clockwise rotation.",
    8003: "The XPL + λ-Plate movie appears to use the wrong rotation direction. Clockwise stage rotation is not allowed. Re-shoot the movies with counter-clockwise rotation.",
    8004: "The XPL movie may not cover enough rotation. Rotate the stage more than 360° and try again.",
    8005: "The XPL + λ-Plate movie may not cover enough rotation. Rotate the stage more than 360° and try again.",
    9001: "An unexpected error occurred while splitting the movie. Reload the video and try again.",
    9002: "An unexpected error occurred while calculating the stage rotation center. Verify the input data.",
    9003: "An unexpected error occurred while calculating rotation angles between images. Restart the process from the beginning.",
    9004: "An unexpected error occurred while calculating the retardation or extinction angle map. Restart the process from the beginning.",
    9999: "An unexpected system error occurred."
}


def get_error_from_error_list(error_no: int) -> str:
    msg = ERROR_MESSAGE.get(error_no)
    if msg is not None:
        return f"E{error_no}: {msg}"
    else:
        raise ValueError("Error code does not exist.")


def exec_at_error(error_no: int, stores: Stores, *, logger: Logger) -> None:
    msg = get_error_from_error_list(error_no)
    print(traceback.format_exc())
    update_progress_bar(0.0, stores)
    update_logs(stores, (msg, "err"))
    traceback.print_exc()
    logger.error(f"displayed: {msg}")
    logger.error(f"traceback: {traceback.format_exc()}")


# %%
if __name__ == "__main__":

    import traceback

    try:
        1 / 0
    except Exception as e:
        print(traceback.format_exc())
