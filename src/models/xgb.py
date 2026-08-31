"""XGBoost 탐지 모델 학습.

설정은 `config/model.yaml`에서만 온다. 코드에 숫자를 박으면 결과 파일만 보고는
어떤 설정으로 나온 값인지 알 수 없다.

**이 함수는 검증셋을 안 받는다.** 학습셋을 시간순으로 둘로 갈라, 앞쪽으로 나무를 그리고
뒤쪽으로 언제 멈출지 정한다. 검증셋은 운영 임계값 τ를 정하는 데만 쓴다.

처음에는 조기 종료도 검증셋으로 했는데, 그러면 τ를 정하는 데이터가 이미 모델을 고르는 데
쓰인 셈이 된다. 나무 수를 잘라가며 재보니 평가셋 PR-AUC는 800그루에서 꼭대기를 찍고
내려가는데 검증셋은 끝까지 올라 1,586그루까지 갔다. 그만큼 검증셋 숫자가 부풀려져 있었다.
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
    fit_rows: int
    # 나무 수를 정하는 데 쓴 행 수. 학습셋 전체로 다시 학습한 모델은 수를 밖에서 받으므로 0이다.
    stop_rows: int
    # "early_stop"이면 조기 종료가 정했고, "fixed"면 밖에서 받았다.
    tree_source: str

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """사기일 확률을 돌려준다. 0/1 판정은 여기서 하지 않는다 — τ는 밖에서 정한다."""
        if list(X.columns) != list(self.feature_names):
            raise ValueError(
                "컬럼 구성이 학습 때와 다릅니다. 전처리를 같은 Preprocessor로 걸었는지 확인하세요."
            )
        # best_iteration까지만 쓴다. 그 뒤 나무는 검증셋에서 더 나빠진 구간이다.
        return self.model.predict_proba(X)[:, 1]

    def save(self, path) -> int:
        """쓸 나무만 잘라서 저장한다. 몇 그루를 남겼는지 돌려준다.

        조기 종료는 최선을 찍은 **뒤에도** `early_stopping_rounds`만큼 더 그려보고 멈춘다.
        그래서 학습이 끝난 booster에는 안 쓸 나무가 그만큼 더 들어 있다. 실제로 첫 실행에서
        1,686그루가 남았는데 쓰는 것은 1,586그루였다.

        `XGBClassifier.predict_proba`는 알아서 잘라 쓰지만, 저장한 파일을 `Booster`로 직접
        불러 `predict()`를 부르면 안 쓸 나무까지 다 쓴다. 그러면 같은 파일인데 부르는 방법에
        따라 점수가 달라진다(평가셋 Recall이 0.6001 대 0.5967로 갈렸다). 공격 단계에서 이
        파일을 불러 쓸 것이라, 파일 자체를 잘라두어야 어느 쪽으로 불러도 같은 값이 나온다.
        """
        keep = self.best_iteration + 1
        self.model.get_booster()[:keep].save_model(str(path))
        return keep


def _base_params(config: dict) -> dict:
    """config에서 설정을 꺼내고, 재현에 필요한 값을 못 박는다."""
    params = dict(config["xgboost"])
    params["random_state"] = config["seed"]
    # 결과가 매번 같게 나와야 한다. 스레드 수가 바뀌면 부동소수점 합산 순서가 달라져
    # 미세하게 결과가 흔들리므로 여기서 못 박는다.
    params.setdefault("n_jobs", 8)
    params.setdefault("objective", "binary:logistic")
    return params


def train_xgb(
    X_fit: pd.DataFrame,
    y_fit,
    X_stop: pd.DataFrame,
    y_stop,
    config: dict,
) -> TrainedModel:
    """앞쪽으로 나무를 그리고 뒤쪽으로 멈출 시점을 정한다. 둘 다 학습셋에서 나온다.

    검증셋과 평가셋은 이 함수에 들어오지 않는다. 넘길 자리가 없어야 실수로 넣지 못한다.
    """
    if list(X_fit.columns) != list(X_stop.columns):
        raise ValueError("두 조각의 컬럼이 다릅니다.")
    if np.asarray(y_fit).sum() == 0:
        raise ValueError("학습 조각에 사기 거래가 없습니다.")
    if np.asarray(y_stop).sum() == 0:
        # 멈출 시점을 정할 근거가 없으면 조기 종료가 아무 데서나 걸린다.
        raise ValueError("조기 종료용 조각에 사기 거래가 없습니다.")

    params = _base_params(config)
    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit, eval_set=[(X_stop, y_stop)], verbose=False)

    return TrainedModel(
        model=model,
        params=params,
        best_iteration=int(getattr(model, "best_iteration", params["n_estimators"] - 1)),
        feature_names=tuple(X_fit.columns),
        fit_rows=len(X_fit),
        stop_rows=len(X_stop),
        tree_source="early_stop",
    )


def refit_on_all(X_train: pd.DataFrame, y_train, n_estimators: int, config: dict) -> TrainedModel:
    """나무 수를 고정해 학습셋 전체로 다시 학습한다. 조기 종료를 쓰지 않는다.

    조기 종료용 조각을 학습셋에서 떼면 나무를 그리는 데 쓸 행이 그만큼 준다. 실제로
    62,006행(15%)이 빠지자 평가셋 PR-AUC가 0.5468에서 0.5116으로 떨어졌다. 나무 수만
    정하고 전체로 다시 학습하면 그 손실이 사라진다 — 같은 720그루로 전체 학습하니
    0.5471로 돌아왔고, ROC-AUC는 0.8984에서 0.9051로 오히려 올랐다. 1,586그루는 이미
    과한 자리였기 때문이다.

    **여기에 검증셋을 넘기면 안 된다.** 나무 수는 이미 정해져 있으므로 검증셋이 할 일이
    없고, 넘기면 τ를 정할 데이터가 학습에 쓰인 것이 된다.

    나무 수를 351,372행에서 정해 413,378행에 그대로 쓰는 셈이라, 데이터가 늘면 최적
    나무 수도 늘 수 있다. 비례해서 847그루로 늘려봤더니 PR-AUC 0.5478, ROC 0.9042로
    거의 같았다. 이 자리에서는 나무 수에 민감하지 않다.
    """
    if n_estimators < 1:
        raise ValueError(f"나무 수가 잘못됐습니다: {n_estimators}")
    if np.asarray(y_train).sum() == 0:
        raise ValueError("학습셋에 사기 거래가 없습니다.")

    params = _base_params(config)
    params["n_estimators"] = n_estimators
    # 조기 종료를 쓰지 않으므로 관련 설정을 뺀다. 남겨두면 eval_set 없이 학습이 막힌다.
    params.pop("early_stopping_rounds", None)

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    return TrainedModel(
        model=model,
        params=params,
        best_iteration=n_estimators - 1,
        feature_names=tuple(X_train.columns),
        fit_rows=len(X_train),
        stop_rows=0,
        tree_source="fixed",
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
