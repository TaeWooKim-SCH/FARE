"""Random Forest 탐지 모델 학습.

XGBoost와 같은 데이터, 같은 분할, 같은 전처리를 쓴다. 다른 것은 나무를 쌓는 방식뿐이다 —
부스팅은 앞 나무가 틀린 것을 다음 나무가 메우고, 배깅은 서로 독립인 나무를 평균 낸다.
전이성 실험에서 이 차이가 "같은 트리 계열 안에서도 공격이 옮겨가나"에 답한다.

**조기 종료가 없다.** 배깅은 나무를 더 쌓아도 과적합하지 않고 평평해지기만 하므로,
XGBoost처럼 학습셋을 갈라 멈출 시점을 찾을 필요가 없다. 나무 수는 config에서 고정하고
학습셋 전체를 한 번에 받는다. 그래서 이 함수도 검증셋을 넘길 자리가 없다.

**결측을 채우지 않는다.** sklearn 1.4부터 RandomForest가 NaN을 직접 다룬다(설치본
1.9.0에서 확인). 채우면 XGBoost와 입력이 달라져, 전이성 결과에서 학습 방식 차이와 입력
차이가 섞인다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass(frozen=True)
class TrainedForest:
    """학습된 숲과, 그 결과를 다시 만드는 데 필요한 기록."""

    model: RandomForestClassifier
    params: dict
    feature_names: tuple[str, ...]
    fit_rows: int

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """사기일 확률을 돌려준다. 0/1 판정은 여기서 하지 않는다 — τ는 밖에서 정한다.

        숲의 확률은 리프에 담긴 사기 비율을 나무마다 구해 평균한 값이다. 부스팅이 내는
        확률과 만들어지는 방식이 다르므로 두 모델의 점수를 같은 자에 놓고 비교하면 안 된다.
        모델마다 τ를 따로 정하는 이유다.
        """
        if list(X.columns) != list(self.feature_names):
            raise ValueError(
                "컬럼 구성이 학습 때와 다릅니다. 전처리를 같은 Preprocessor로 걸었는지 확인하세요."
            )
        return self.model.predict_proba(X)[:, 1]

    def save(self, path) -> int:
        """숲을 파일로 남기고 바이트 크기를 돌려준다.

        XGBoost는 안 쓸 나무를 잘라내야 했지만 숲은 그럴 것이 없다 — 모든 나무를 다 쓴다.
        압축을 거는 이유는 나무 하나가 수 MB라 200그루면 파일이 수백 MB가 되기 때문이다.
        """
        path = Path(path)
        joblib.dump(self.model, path, compress=3)
        return path.stat().st_size


def _base_params(config: dict) -> dict:
    """config에서 설정을 꺼내고, 재현에 필요한 값을 못 박는다.

    여기 담기는 `n_jobs`는 **학습에 쓴 값**이다. 예측은 `train_rf`가 1로 바꿔 고정한다.
    """
    params = dict(config["random_forest"])
    params["random_state"] = config["seed"]
    params.setdefault("n_jobs", 8)
    return params


def train_rf(X_train: pd.DataFrame, y_train, config: dict) -> TrainedForest:
    """학습셋 전체로 숲을 기른다.

    **검증셋과 평가셋은 이 함수에 들어오지 않는다.** 넘길 자리가 없어야 실수로 넣지 못한다.
    """
    if np.asarray(y_train).sum() == 0:
        raise ValueError("학습셋에 사기 거래가 없습니다.")

    params = _base_params(config)
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # 학습은 병렬로 하되 **예측은 한 스레드로 못 박는다.** 나무는 병렬로 길러도 똑같이
    # 나오지만(각 나무의 시드를 순서대로 미리 뽑아 쓰므로), 확률을 낼 때는 나무별 결과를
    # 여러 스레드가 하나의 배열에 더해 넣는다. 부동소수점 덧셈은 순서를 타므로 같은 모델로
    # 같은 입력을 두 번 넣어도 마지막 비트가 흔들린다. 실제로 n_jobs=8에서 세 번 부르면
    # 세 번 다 달랐고, 1로 두니 비트까지 같아졌다.
    #
    # 공격 단계에서 τ를 경계로 넘나드는지를 보는데, 그 경계에 걸친 거래는 이 정도 흔들림에도
    # 판정이 뒤집힌다. 예측 속도보다 재현성이 중요한 자리다.
    model.n_jobs = 1

    return TrainedForest(
        model=model,
        params=params,
        feature_names=tuple(X_train.columns),
        fit_rows=len(X_train),
    )


def feature_importance(trained: TrainedForest, top: int = 20) -> pd.DataFrame:
    """숲이 어느 컬럼에 기댔는지 본다. **값을 곧이곧대로 읽으면 안 된다.**

    sklearn이 주는 `feature_importances_`는 불순도 감소량(MDI)인데, 값 종류가 많은 컬럼을
    과대평가하는 성질이 있다. 자를 자리가 많으면 학습셋을 우연히 잘 가르는 자리도 많아지기
    때문이다. 이 데이터에는 card1(12,242종)·card2(500종)·addr1(318종)처럼 종류가 많은
    컬럼이 있어서 이 편향이 실제로 걸린다.

    그래서 이 값은 "숲이 무엇을 봤나"의 참고용이고, "그 컬럼이 없으면 얼마나 나빠지나"는
    아니다. 그건 값을 섞어보는 순열 중요도로만 나오고, 필요한 자리에서 따로 잰다.
    XGBoost 쪽 `total_gain`과도 만들어지는 방식이 달라 순위를 직접 비교하면 안 된다.
    """
    frame = pd.DataFrame(
        {
            "feature": list(trained.feature_names),
            "mdi": trained.model.feature_importances_,
        }
    )
    return frame.sort_values("mdi", ascending=False, ignore_index=True).head(top)
