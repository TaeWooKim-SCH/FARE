"""MLP 탐지 모델 학습.

XGBoost·Random Forest와 같은 데이터, 같은 분할, 같은 임계값 절차를 쓴다. 다른 것은 입력
변환(`mlp_adapter.py`)과 경계의 모양이다.

**MLP를 넣는 이유는 제일 센 탐지기여서가 아니라 제일 다른 경계여서다.** 트리는 축에 나란한
계단 함수를 그린다 — 금액을 조금 바꿔도 분기 기준값을 안 넘으면 점수가 정확히 0만큼
움직이고, 넘는 순간 뚝 떨어진다. MLP는 연속 함수라 조금 바꾸면 조금 움직인다. 같은 공격을
걸어도 성격이 달라지므로, 전이성 실험에서 "공격이 모델 구조와 무관한가"에 답할 수 있다
(research-plan.md 4.1 ③). 정형 데이터에서 트리보다 점수가 낮게 나올 수 있고, 그것은
결함이 아니라 예상된 결과다. 논문에 이 위치를 명시한다.

**sklearn의 `early_stopping=True`를 쓰지 않는다.** 켜면 내부에서
`train_test_split(X, y, random_state=..., test_size=validation_fraction)`으로 검증
조각을 떼는데, **이것이 무작위 분할이다.** 시간이 섞여 미래가 과거로 새므로 이 연구의
절대 규칙 1을 정면으로 어긴다. 코드에 아무 표시도 안 나고 조용히 일어난다.

대신 두 가지로 학습을 멈춘다.

- `max_iter` — 몇 바퀴 돌지. 하이퍼파라미터로 두고 다음 PR의 탐색에서 정한다
- `n_iter_no_change` + `tol` — **학습 손실**이 더 안 줄면 멈춘다. 검증 조각을 안 쓰므로
  분할을 건드리지 않는다

XGBoost처럼 학습셋 뒤쪽 조각으로 멈출 시점을 찾는 방법도 생각했는데 접었다. sklearn MLP는
eval_set을 받지 않아서 `warm_start`로 몇 바퀴씩 끊어 돌며 재야 하는데, 끊을 때마다 Adam의
관성이 초기화되어 이어서 도는 것과 결과가 달라진다. 멈출 지점을 재려다 학습 자체를 바꾸는
셈이라, `max_iter`를 탐색 대상으로 넘기는 쪽이 정직하다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier


@dataclass(frozen=True)
class TrainedNetwork:
    """학습된 신경망과, 그 결과를 다시 만드는 데 필요한 기록."""

    model: MLPClassifier
    params: dict
    feature_names: tuple[str, ...]
    fit_rows: int
    # 실제로 돈 바퀴 수와 마지막 학습 손실. max_iter에 닿았는지 손실이 평평해져 멈췄는지
    # 결과 파일만 보고 알 수 있어야 한다.
    rounds: int
    final_loss: float

    @property
    def stopped_early(self) -> bool:
        """손실이 평평해져 멈췄으면 True. max_iter를 다 썼으면 False다."""
        return self.rounds < self.params["max_iter"]

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """사기일 확률을 돌려준다. 0/1 판정은 여기서 하지 않는다 — τ는 밖에서 정한다."""
        if list(X.columns) != list(self.feature_names):
            raise ValueError(
                "컬럼 구성이 학습 때와 다릅니다. 같은 MlpAdapter를 거쳤는지 확인하세요."
            )
        return self.model.predict_proba(X)[:, 1].astype("float64")

    def save(self, path) -> int:
        """신경망을 파일로 남기고 바이트 크기를 돌려준다."""
        path = Path(path)
        joblib.dump(self.model, path, compress=3)
        return path.stat().st_size


def _base_params(config: dict) -> dict:
    """config에서 설정을 꺼내고, 재현에 필요한 값을 못 박는다."""
    params = dict(config["mlp"]["network"])
    params["random_state"] = config["seed"]
    # yaml은 리스트로 주는데 sklearn은 튜플을 기대한다. 리스트로 넘기면 경고가 난다.
    params["hidden_layer_sizes"] = tuple(params["hidden_layer_sizes"])
    # 무작위 분할이라 절대 켜지 않는다. config에서 실수로 켜도 여기서 막는다.
    params["early_stopping"] = False
    return params


def train_mlp(X_train: pd.DataFrame, y_train, config: dict) -> TrainedNetwork:
    """학습셋 전체로 신경망을 학습한다.

    **검증셋과 평가셋은 이 함수에 들어오지 않는다.** 넘길 자리가 없어야 실수로 넣지 못한다.

    입력을 float32로 넘기면 sklearn이 그대로 float32로 학습한다(가중치도 점수도). 41만 행
    543컬럼이면 float64로 올라갈 때 1.8GB인데 float32면 898MB라 차이가 크다.
    `MlpAdapter`가 float32로 내주므로 여기서 다시 만지지 않는다.
    """
    if np.asarray(y_train).sum() == 0:
        raise ValueError("학습셋에 사기 거래가 없습니다.")

    params = _base_params(config)
    model = MLPClassifier(**params)
    model.fit(X_train, y_train)

    return TrainedNetwork(
        model=model,
        params=params,
        feature_names=tuple(X_train.columns),
        fit_rows=len(X_train),
        rounds=int(model.n_iter_),
        final_loss=float(model.loss_),
    )


def loss_curve(trained: TrainedNetwork) -> pd.DataFrame:
    """바퀴별 학습 손실. 덜 돌았는지 과하게 돌았는지 보는 용도다.

    **이것은 성능 곡선이 아니다.** 학습셋 손실이라 계속 내려가는 것이 정상이고, 내려간다고
    좋아지는 것도 아니다. `max_iter`에 닿아 멈췄는지 손실이 평평해져 멈췄는지를 가르는 데
    쓰고, 실제 성능은 검증셋 PR-AUC로 본다.

    트리처럼 "어느 컬럼을 몇 번 썼나"에 해당하는 값은 신경망에 없다. 컬럼 기여도를 보려면
    값을 섞어보는 순열 중요도가 필요한데 543컬럼마다 다시 예측해야 해서 비싸다. 필요한
    자리에서 따로 잰다.
    """
    curve = list(trained.model.loss_curve_)
    return pd.DataFrame({"round": range(1, len(curve) + 1), "loss": curve})
