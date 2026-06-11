import numpy as np
from typing import Optional, List
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
        kernel: str = "knn",  # "rbf" も可
        gamma: float = 20.0,  # kernel="rbf" 用
        use_robust_scaler: bool = True,
        reject_threshold: float = 0.55,  # 最大クラス確率がこれ未満なら class 0
        position_weight: float = 0.2,
        prototype_weight: float = 0.85,
    ):
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.gamma = gamma
        self.use_robust_scaler = use_robust_scaler
        self.reject_threshold = reject_threshold
        self.position_weight = position_weight
        self.prototype_weight = prototype_weight

        self.X_: Optional[np.ndarray] = None  # (N, d) 生特徴
        self.scaler_ = None
        self.Z_: Optional[np.ndarray] = None  # (N, d) スケール後
        self.y_user_: Optional[np.ndarray] = None  # (N,) 学習用。未ラベルは -1
        self.lp_: Optional[LabelPropagation] = None

        self.classes_in_use_: List[int] = []  # 0 を除く使用中クラス（昇順）
        self.y_pred_: Optional[np.ndarray] = None  # (N,) 予測（拒否規則適用後）
        self.y_proba_: Optional[np.ndarray] = None  # (N, C) 予測確率
        self.proba_classes_: Optional[np.ndarray] = None

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
        if self.Z_.shape[1] >= 9:
            self.Z_[:, -2:] *= float(self.position_weight)
        N = self.Z_.shape[0]
        # 最初はすべて未ラベル（-1）。推論の既定値は後段で 0 とする
        self.y_user_ = np.full(N, -1, dtype=int)
        self.classes_in_use_ = []
        # 初期状態の予測は全部 0
        self.y_pred_ = np.zeros(N, dtype=int)
        self.y_proba_ = None
        self.proba_classes_ = None
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

    def clear_class(self, class_id: int) -> None:
        """指定クラスに付与されたユーザーラベルをすべて未ラベルへ戻す。"""
        if self.y_user_ is None:
            return
        mask = self.y_user_ == class_id
        if not np.any(mask):
            return
        self.y_user_[mask] = -1
        if class_id in self.classes_in_use_:
            self.classes_in_use_.remove(class_id)

    # -------------------- 伝播と予測 --------------------
    def propagate(self) -> None:
        """
        現在のユーザーラベル y_user_（-1: 未ラベル）で LabelPropagation を実行し、
        低確信を class 0 へ落として y_pred_ を更新。
        """
        if self.Z_ is None or self.y_user_ is None:
            raise RuntimeError("No features/labels. Call fit_features and set_label.")

        labeled_mask = self.y_user_ > 0
        if labeled_mask.sum() == 0:
            self.y_pred_ = np.zeros_like(self.y_user_, dtype=int)
            self.y_proba_ = None
            self.proba_classes_ = None
            self.lp_ = None
            return

        prototype_proba, classes = self._prototype_probabilities(labeled_mask)
        graph_proba = self._label_propagation_probabilities(classes)
        if graph_proba is None:
            proba = prototype_proba
        else:
            alpha = float(np.clip(self.prototype_weight, 0.0, 1.0))
            proba = alpha * prototype_proba + (1.0 - alpha) * graph_proba
            proba = self._normalize_rows(proba)

        argmax_idx = proba.argmax(axis=1)
        max_prob = proba[np.arange(proba.shape[0]), argmax_idx]
        y_pred = classes[argmax_idx]
        y_pred[max_prob < self.reject_threshold] = 0
        y_pred[self.y_user_ > 0] = self.y_user_[self.y_user_ > 0]

        self.y_pred_ = y_pred.astype(int)
        self.y_proba_ = proba
        self.proba_classes_ = classes

    def _prototype_probabilities(
        self,
        labeled_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.Z_ is None or self.y_user_ is None:
            raise RuntimeError("No features/labels. Call fit_features and set_label.")

        classes = np.array(sorted(set(self.y_user_[labeled_mask].tolist())), dtype=int)
        labeled_indices = np.flatnonzero(labeled_mask)
        labeled_z = self.Z_[labeled_indices]
        diff = self.Z_[:, None, :] - labeled_z[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        min_dist = np.sqrt(np.min(dist2, axis=1))
        positive_dist = min_dist[min_dist > 0]
        if positive_dist.size:
            sigma = float(np.median(positive_dist))
        else:
            sigma = 1.0
        sigma = max(sigma, 1e-6)

        weights = np.exp(-0.5 * dist2 / (sigma * sigma))
        scores = np.zeros((self.Z_.shape[0], classes.size), dtype=np.float64)
        for col, class_id in enumerate(classes):
            class_cols = self.y_user_[labeled_indices] == class_id
            if not np.any(class_cols):
                continue
            class_weights = weights[:, class_cols]
            scores[:, col] = np.max(class_weights, axis=1)

        for row, class_id in zip(labeled_indices, self.y_user_[labeled_indices]):
            class_pos = np.flatnonzero(classes == class_id)
            if class_pos.size:
                scores[row, :] = 0.0
                scores[row, class_pos[0]] = 1.0

        return self._normalize_rows(scores), classes

    def _label_propagation_probabilities(
        self,
        classes: np.ndarray,
    ) -> Optional[np.ndarray]:
        if self.Z_ is None or self.y_user_ is None:
            return None
        if classes.size == 0:
            return None

        kwargs = dict(kernel=self.kernel, max_iter=1000)
        if self.kernel == "knn":
            N = self.Z_.shape[0]
            nnb = max(1, min(self.n_neighbors, N - 1))
            kwargs["n_neighbors"] = nnb
        else:
            kwargs["gamma"] = float(self.gamma)

        self.lp_ = LabelPropagation(**kwargs)
        try:
            self.lp_.fit(self.Z_, self.y_user_)
        except Exception:
            self.lp_ = None
            return None

        lp_classes = np.asarray(self.lp_.classes_, dtype=int)
        lp_proba = np.asarray(self.lp_.label_distributions_, dtype=np.float64)
        out = np.zeros((self.Z_.shape[0], classes.size), dtype=np.float64)
        for col, class_id in enumerate(classes):
            match = np.flatnonzero(lp_classes == class_id)
            if match.size:
                out[:, col] = lp_proba[:, match[0]]
        return self._normalize_rows(out)

    @staticmethod
    def _normalize_rows(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=np.float64)
        totals = scores.sum(axis=1, keepdims=True)
        empty = totals[:, 0] <= 0
        totals[empty, 0] = 1.0
        out = scores / totals
        if np.any(empty) and out.shape[1] > 0:
            out[empty, :] = 1.0 / out.shape[1]
        return out

    # -------------------- 出力 --------------------
    def current_predictions(self) -> np.ndarray:
        """拒否規則込みの現在の予測（未伝播なら全 0）。"""
        if self.y_pred_ is None:
            return np.zeros_like(self.y_user_, dtype=int)
        return self.y_pred_.copy()

    def current_probabilities(self) -> Optional[np.ndarray]:
        """LabelPropagation の分布（未伝播なら None）。"""
        return None if self.y_proba_ is None else self.y_proba_.copy()

    def probability_classes(self) -> Optional[np.ndarray]:
        return None if self.proba_classes_ is None else self.proba_classes_.copy()

    def labeled_mask(self) -> np.ndarray:
        """ユーザーが明示ラベルを付けた位置（-1 以外）。"""
        return self.y_user_ != -1

    def classes_in_use(self) -> List[int]:
        """0 を除いた使用中クラス一覧（昇順）。"""
        return list(self.classes_in_use_)
