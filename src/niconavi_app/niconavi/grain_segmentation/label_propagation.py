import numpy as np
from typing import Optional, Tuple, List
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.semi_supervised import LabelPropagation

class InteractiveLabelPropagation:
    """
    対話的ラベル付け + LabelPropagation による類似拡張。
    - 背景/その他のクラス = 0（予測の既定値）
    - 学習用の未ラベルは -1
    - 伝播後、確信度が低い点は 0 へ落とす（reject_threshold）
    """
    def __init__(
        self,
        n_neighbors: int = 10,
        kernel: str = "knn",                 # "rbf" も可
        gamma: float = 20.0,                 # kernel="rbf" 用
        use_robust_scaler: bool = True,
        reject_threshold: float = 0.55       # 最大クラス確率がこれ未満なら class 0
    ):
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.gamma = gamma
        self.use_robust_scaler = use_robust_scaler
        self.reject_threshold = reject_threshold

        self.X_: Optional[np.ndarray] = None     # (N, d) 生特徴
        self.scaler_ = None
        self.Z_: Optional[np.ndarray] = None     # (N, d) スケール後
        self.y_user_: Optional[np.ndarray] = None  # (N,) 学習用。未ラベルは -1
        self.lp_: Optional[LabelPropagation] = None

        self.classes_in_use_: List[int] = []     # 0 を除く使用中クラス（昇順）
        self.y_pred_: Optional[np.ndarray] = None  # (N,) 予測（拒否規則適用後）
        self.y_proba_: Optional[np.ndarray] = None # (N, C) 予測確率（LP の分布）

    # -------------------- データ/特徴 --------------------
    def fit_features(self, X: np.ndarray) -> None:
        """
        X: (N, d) 特徴行列（例: [logA, C, solidity, aspect_ratio, Lab(a), Lab(b), HSV(S)])
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array.")
        self.X_ = X.astype(np.float64, copy=True)
        Scaler = RobustScaler if self.use_robust_scaler else StandardScaler
        self.scaler_ = Scaler().fit(self.X_)
        self.Z_ = self.scaler_.transform(self.X_)
        N = self.Z_.shape[0]
        # 最初はすべて未ラベル（-1）。推論の既定値は後段で 0 とする
        self.y_user_ = np.full(N, -1, dtype=int)
        self.classes_in_use_ = []
        # 初期状態の予測は全部 0
        self.y_pred_ = np.zeros(N, dtype=int)
        self.y_proba_ = None
        self.lp_ = None

    # -------------------- ラベル操作 --------------------
    def set_label(self, idx: int, class_id: Optional[int] = None) -> int:
        """
        ユーザーが「index idx を class_id に」指定。class_id=None なら新クラス払い出し。
        Returns: 確定した class_id
        """
        if self.Z_ is None:
            raise RuntimeError("Call fit_features(X) first.")
        if class_id is None or class_id < 0:
            # 0 は背景用に予約。既存最大 + 1 を払い出し
            next_class = (max(self.classes_in_use_) + 1) if self.classes_in_use_ else 1
            class_id = next_class
        self.y_user_[idx] = int(class_id)
        if class_id != 0 and class_id not in self.classes_in_use_:
            self.classes_in_use_.append(class_id)
            self.classes_in_use_.sort()
        return int(class_id)

    # -------------------- 伝播と予測 --------------------
    def propagate(self) -> None:
        """
        現在のユーザーラベル y_user_（-1: 未ラベル）で LabelPropagation を実行し、
        低確信を class 0 へ落として y_pred_ を更新。
        """
        if self.Z_ is None or self.y_user_ is None:
            raise RuntimeError("No features/labels. Call fit_features and set_label.")

        labeled_mask = (self.y_user_ != -1)
        if labeled_mask.sum() == 0:
            self.y_pred_ = np.zeros_like(self.y_user_, dtype=int)
            self.y_proba_ = None
            self.lp_ = None
            return

        # --- ここを修正：kernel に応じて渡す引数を分ける ---
        kwargs = dict(kernel=self.kernel, max_iter=1000)
        if self.kernel == "knn":
            # n_neighbors は [1, N-1] に収める
            N = self.Z_.shape[0]
            nnb = max(1, min(self.n_neighbors, N - 1))
            kwargs["n_neighbors"] = nnb
            # gamma は渡さない（デフォルト float を維持）
        else:  # "rbf"
            kwargs["gamma"] = float(self.gamma)  # None は渡さない

        self.lp_ = LabelPropagation(**kwargs)
        self.lp_.fit(self.Z_, self.y_user_)

        proba = self.lp_.label_distributions_  # (N, n_classes_)
        classes = self.lp_.classes_            # 観測クラスID（昇順）

        argmax_idx = proba.argmax(axis=1)
        max_prob = proba[np.arange(proba.shape[0]), argmax_idx]
        pred_raw = classes[argmax_idx]

        # 拒否規則：確信が低いものは class 0
        y_pred = pred_raw.copy()
        y_pred[max_prob < self.reject_threshold] = 0

        self.y_pred_ = y_pred.astype(int)
        self.y_proba_ = proba

    # -------------------- 出力 --------------------
    def current_predictions(self) -> np.ndarray:
        """拒否規則込みの現在の予測（未伝播なら全 0）。"""
        if self.y_pred_ is None:
            return np.zeros_like(self.y_user_, dtype=int)
        return self.y_pred_.copy()

    def current_probabilities(self) -> Optional[np.ndarray]:
        """LabelPropagation の分布（未伝播なら None）。"""
        return None if self.y_proba_ is None else self.y_proba_.copy()

    def labeled_mask(self) -> np.ndarray:
        """ユーザーが明示ラベルを付けた位置（-1 以外）。"""
        return (self.y_user_ != -1)

    def classes_in_use(self) -> List[int]:
        """0 を除いた使用中クラス一覧（昇順）。"""
        return list(self.classes_in_use_)