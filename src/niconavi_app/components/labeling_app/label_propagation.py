from typing import List, Optional

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.semi_supervised import LabelPropagation


class InteractiveLabelPropagation:
    """
    対話的ラベル付け + LabelPropagation による類似拡張。

    有効な試料粒子だけで特徴量のk近傍グラフを作り、ユーザーが付けた
    ラベルを固定したまま未ラベル粒子へ伝播する。

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
        scale_features: bool = True,
    ):
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.gamma = gamma
        self.use_robust_scaler = use_robust_scaler
        self.reject_threshold = reject_threshold
        self.position_weight = position_weight
        self.scale_features = scale_features

        self.X_: Optional[np.ndarray] = None  # (N, d) 生特徴
        self.feature_weights_: Optional[np.ndarray] = None
        self.scaler_ = None
        self.Z_: Optional[np.ndarray] = None  # (N, d) スケール後
        self.valid_mask_: Optional[np.ndarray] = None  # (N,) 実在する粒子
        self.y_user_: Optional[np.ndarray] = None  # (N,) 学習用。未ラベルは -1
        self.lp_: Optional[LabelPropagation] = None

        self.classes_in_use_: List[int] = []  # 0 を除く使用中クラス（昇順）
        self.y_pred_: Optional[np.ndarray] = None  # (N,) 予測（拒否規則適用後）
        self.y_proba_: Optional[np.ndarray] = None  # (N, C) 予測確率
        self.proba_classes_: Optional[np.ndarray] = None

    # -------------------- データ/特徴 --------------------
    def fit_features(
        self,
        X: np.ndarray,
        feature_weights: Optional[np.ndarray] = None,
        valid_mask: Optional[np.ndarray] = None,
    ) -> None:
        """
        X: (N, d) feature matrix (for example: elongation, size, Lab, position)
        valid_mask: (N,) mask for rows that correspond to actual grains
        """
        if X.ndim != 2:
            raise ValueError("X must be 2D array.")
        self.X_ = X.astype(np.float64, copy=True)
        N = self.X_.shape[0]
        if valid_mask is None:
            self.valid_mask_ = np.ones(N, dtype=bool)
        else:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.ndim != 1 or mask.size != N:
                raise ValueError("valid_mask must match the feature rows.")
            self.valid_mask_ = mask.copy()
        if feature_weights is None:
            weights = np.ones(X.shape[1], dtype=np.float64)
            if weights.size >= 9:
                weights[-2:] = float(self.position_weight)
        else:
            weights = np.asarray(feature_weights, dtype=np.float64)
            if weights.ndim != 1 or weights.size != X.shape[1]:
                raise ValueError("feature_weights must match the feature columns.")
        selected = weights > 0
        if not np.any(selected):
            selected = np.ones(X.shape[1], dtype=bool)
            weights = np.ones(X.shape[1], dtype=np.float64)
            if weights.size >= 9:
                weights[-2:] = float(self.position_weight)
        self.feature_weights_ = weights.copy()
        weighted_x = self.X_[:, selected]
        if self.scale_features and np.any(self.valid_mask_):
            Scaler = RobustScaler if self.use_robust_scaler else StandardScaler
            scaler = Scaler().fit(weighted_x[self.valid_mask_])
            self.scaler_ = scaler
            self.Z_ = scaler.transform(weighted_x)
        else:
            self.scaler_ = None
            self.Z_ = weighted_x.copy()
        self.Z_ *= weights[selected]
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
        if self.Z_ is None or self.valid_mask_ is None or self.y_user_ is None:
            raise RuntimeError("Call fit_features(X) first.")
        if idx < 0 or idx >= self.Z_.shape[0]:
            raise IndexError("label index is outside the feature rows.")
        if not self.valid_mask_[idx]:
            raise ValueError("cannot label a row that does not represent a grain.")
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
        if self.Z_ is None or self.y_user_ is None or self.valid_mask_ is None:
            raise RuntimeError("No features/labels. Call fit_features and set_label.")

        labeled_mask = (self.y_user_ > 0) & self.valid_mask_
        if labeled_mask.sum() == 0:
            self.y_pred_ = np.zeros_like(self.y_user_, dtype=int)
            self.y_proba_ = None
            self.proba_classes_ = None
            self.lp_ = None
            return

        classes = np.array(sorted(set(self.y_user_[labeled_mask].tolist())), dtype=int)
        proba = self._label_propagation_probabilities(classes)

        y_pred = np.zeros_like(self.y_user_, dtype=int)
        propagated_mask = self.valid_mask_ & (proba.sum(axis=1) > 0)
        if np.any(propagated_mask):
            rows = np.flatnonzero(propagated_mask)
            argmax_idx = proba[rows].argmax(axis=1)
            max_prob = proba[rows, argmax_idx]
            accepted = max_prob >= self.reject_threshold
            y_pred[rows[accepted]] = classes[argmax_idx[accepted]]
        y_pred[labeled_mask] = self.y_user_[labeled_mask]

        self.y_pred_ = y_pred.astype(int)
        self.y_proba_ = proba
        self.proba_classes_ = classes

    def _label_propagation_probabilities(
        self,
        classes: np.ndarray,
    ) -> np.ndarray:
        if self.Z_ is None or self.y_user_ is None or self.valid_mask_ is None:
            raise RuntimeError("No features/labels. Call fit_features and set_label.")

        valid_indices = np.flatnonzero(self.valid_mask_)
        out = np.zeros((self.Z_.shape[0], classes.size), dtype=np.float64)
        if classes.size == 0 or valid_indices.size == 0:
            return out

        if self.kernel == "knn":
            nnb = max(1, min(self.n_neighbors, valid_indices.size))

            def symmetric_knn_kernel(x: np.ndarray, _y: np.ndarray):
                connectivity = kneighbors_graph(
                    x,
                    n_neighbors=nnb,
                    mode="connectivity",
                    include_self=True,
                )
                # sklearn's built-in kNN kernel is directed.  A symmetric
                # affinity lets a label propagate naturally in either
                # direction along a neighborhood connection.
                connectivity = connectivity.maximum(connectivity.T).tocoo()
                delta = x[connectivity.row] - x[connectivity.col]
                distances = np.sqrt(np.sum(delta * delta, axis=1))
                positive_distances = distances[distances > 0]
                bandwidth = (
                    float(np.median(positive_distances))
                    if positive_distances.size
                    else 1.0
                )
                bandwidth = max(bandwidth, float(np.finfo(np.float64).eps))
                affinities = np.exp(-0.5 * (distances / bandwidth) ** 2)
                return csr_matrix(
                    (affinities, (connectivity.row, connectivity.col)),
                    shape=connectivity.shape,
                )

            self.lp_ = LabelPropagation(
                kernel=symmetric_knn_kernel,
                max_iter=1000,
            )
        elif self.kernel == "rbf":
            self.lp_ = LabelPropagation(
                kernel="rbf",
                gamma=float(self.gamma),
                max_iter=1000,
            )
        else:
            raise ValueError(f"unsupported LabelPropagation kernel: {self.kernel!r}")

        self.lp_.fit(self.Z_[valid_indices], self.y_user_[valid_indices])

        lp_classes = np.asarray(self.lp_.classes_, dtype=int)
        lp_proba = np.asarray(self.lp_.label_distributions_, dtype=np.float64)
        for col, class_id in enumerate(classes):
            match = np.flatnonzero(lp_classes == class_id)
            if match.size:
                out[valid_indices, col] = lp_proba[:, match[0]]

        totals = out.sum(axis=1, keepdims=True)
        nonempty = totals[:, 0] > 0
        out[nonempty] /= totals[nonempty]
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
        if self.y_user_ is None:
            raise RuntimeError("Call fit_features(X) first.")
        return self.y_user_ != -1

    def classes_in_use(self) -> List[int]:
        """0 を除いた使用中クラス一覧（昇順）。"""
        return list(self.classes_in_use_)
