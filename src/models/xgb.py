"""XGBoost 탐지 모델 학습.

설정은 `config/model.yaml`에서만 온다. 코드에 숫자를 박으면 결과 파일만 보고는
어떤 설정으로 나온 값인지 알 수 없다.

**검증셋을 두 번 쓴다.** 학습을 언제 멈출지 정하는 데 쓰고, 임계값 τ를 정하는 데도 쓴다.
그래서 검증셋 성능은 실제보다 좋게 나온다. 평가셋은 둘 중 어디에도 안 쓰므로 평가셋
숫자만 보고한다. 논문에도 이 순서를 적는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from xgboost import XGBClassifier


@dataclass(frozen=True)
class TrainedModel:
    """학습된 모델과, 그 결과를 다시 만드는 데 필요한 기록."""

    model: XGBClassifier
    params: dict
    best_iteration: int
    feature_names: tuple[str, ...]
    train_rows: int
    val_rows: int

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """사기일 확률을 돌려준다. 0/1 판정은 여기서 하지 않는다 — τ는 밖에서 정한다."""
        if list(X.columns) != list(self.feature_names):
            raise ValueError(
                "컬럼 구성이 학습 때와 다릅니다. 전처리를 같은 Preprocessor로 걸었는지 확인하세요."
            )
        # best_iteration까지만 쓴다. 그 뒤 나무는 검증셋에서 더 나빠진 구간이다.
        return self.model.predict_proba(X)[:, 1]


def train_xgb(
    X_train: pd.DataFrame,
    y_train,
    X_val: pd.DataFrame,
    y_val,
    config: dict,
) -> TrainedModel:
    """학습셋으로 학습하고 검증셋으로 멈출 시점을 정한다.

    평가셋은 이 함수에 들어오지 않는다. 넘길 자리가 없어야 실수로 넣지 못한다.
    """
    if list(X_train.columns) != list(X_val.columns):
        raise ValueError("학습셋과 검증셋의 컬럼이 다릅니다.")
    if np.asarray(y_train).sum() == 0:
        raise ValueError("학습셋에 사기 거래가 없습니다.")

    params = dict(config["xgboost"])
    params["random_state"] = config["seed"]
    # 결과가 매번 같게 나와야 한다. 스레드 수가 바뀌면 부동소수점 합산 순서가 달라져
    # 미세하게 결과가 흔들리므로 여기서 못 박는다.
    params.setdefault("n_jobs", 8)
    params.setdefault("objective", "binary:logistic")

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return TrainedModel(
        model=model,
        params=params,
        best_iteration=int(getattr(model, "best_iteration", params["n_estimators"] - 1)),
        feature_names=tuple(X_train.columns),
        train_rows=len(X_train),
        val_rows=len(X_val),
    )


def feature_importance(trained: TrainedModel, top: int = 20) -> pd.DataFrame:
    """컬럼을 얼마나 자주 썼고(weight) 총 기여가 얼마인지(total_gain) 함께 낸다.

    **`importance_type="gain"`만 보면 안 된다.** xgboost의 `gain`은 총합이 아니라
    분기 하나당 평균이라, 딱 한 번 쓰고 크게 갈라진 컬럼이 1등으로 올라온다. 실제로
    V258은 `gain` 18.65%로 1등인데 분기 횟수는 0.09%뿐이고, 평가셋에서 그 값을 섞어도
    PR-AUC가 0.0059밖에 안 떨어진다. 같은 V 블록에 대체재가 많아 모델이 우회한다.

    두 지표를 같이 봐도 "그 컬럼이 없으면 얼마나 나빠지나"는 아니다. 그건 값을 섞어보는
    순열 중요도로만 나오고, 컬럼마다 다시 예측해야 해서 비싸다. 필요한 자리에서 따로 잰다.
    """
    booster = trained.model.get_booster()
    weight = booster.get_score(importance_type="weight")
    total_gain = booster.get_score(importance_type="total_gain")

    frame = pd.DataFrame(
        [
            {
                "feature": f,
                "splits": int(weight.get(f, 0)),
                "total_gain": float(total_gain.get(f, 0.0)),
            }
            for f in trained.feature_names
        ]
    )
    for column in ("splits", "total_gain"):
        total = frame[column].sum()
        frame[f"{column}_share"] = frame[column] / total if total else 0.0
    return frame.sort_values("total_gain", ascending=False, ignore_index=True).head(top)
