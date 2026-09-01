"""모델별 입력 어댑터.

공통 전처리(`preprocess.py`)는 세 모델이 똑같이 거친다. 그 뒤에 모델마다 다른 처리가
붙는데, 그것을 이 층에서 한다.

**이 층을 따로 두는 이유는 공격 단계 때문이다.** 공격은 원본 거래의 `TransactionAmt`와
`TransactionDT`를 바꾼 뒤 전처리를 다시 태워 점수를 받는다. 모델 3종에 각각 다른 전처리가
직접 박혀 있으면 공격 코드가 세 갈래로 갈라진다. 어댑터로 묶어두면 공격 코드는 원본
프레임만 다루고 세 모델을 같은 방법으로 부를 수 있다.

어댑터가 지키는 약속은 둘뿐이다.

    fit_*(X_train, pre, config) -> 어댑터    규칙은 학습셋에서만 만든다
    어댑터.apply(X) -> DataFrame             만든 규칙을 적용만 한다

`fit_*`의 인자가 셋인 이유는 MLP 어댑터가 그만큼을 필요로 해서다. 공통 전처리를 거치면
컬럼이 전부 정수라서 프레임만 봐서는 ProductCD(코드)와 card3(값)를 구분할 수 없어
`pre`가 필요하고, 원핫 기준과 식별자 목록이 `config`에서 온다. `PassThrough`는 둘 다
안 쓰지만 `runner`가 어댑터를 갈아 끼우려면 모양이 같아야 한다.

XGBoost와 Random Forest는 여기서 아무것도 안 한다(`PassThrough`). 트리에는 필요가 없어서다.

- **스케일링** — 단조 변환은 값의 순서를 안 바꾸는데 트리 분기는 순서만 본다. 걸어도 결과가
  같다.
- **원핫** — `card4 <= 1.5` 같은 분기로 값 집합을 잘라낼 수 있다. 원핫하면 오히려 컬럼이
  희소해져 분기 하나가 가르는 행이 줄고, 같은 정보를 얻으려면 나무가 깊어져야 한다.
- **결측 채우기** — XGBoost는 분기마다 결측을 어느 쪽으로 보낼지 학습에서 정하고,
  sklearn RandomForest도 1.4부터 NaN을 그대로 받는다. 채우면 두 트리의 입력이 서로
  달라져, 전이성 결과에서 학습 방식 차이와 입력 차이가 섞인다.

MLP용 어댑터는 다음 PR에서 붙인다. 거기서는 원핫·빈도 인코딩·분위수 변환·결측 채우기가
전부 들어간다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PassThrough:
    """트리용 어댑터. 값을 손대지 않고 컬럼 구성만 확인한다."""

    feature_columns: tuple[str, ...]

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """그대로 돌려준다.

        아무것도 안 하는데 왜 확인을 하냐면, 컬럼 순서가 어긋나도 모델은 예외를 안 내고
        조용히 엉뚱한 값을 읽기 때문이다. 어댑터가 붙는 자리가 전처리와 모델 사이라
        여기서 잡는 것이 제일 빠르다.
        """
        got = tuple(X.columns)
        if got != self.feature_columns:
            missing = [c for c in self.feature_columns if c not in got]
            raise ValueError(
                "컬럼 구성이 학습 때와 다릅니다"
                + (f" (빠진 컬럼: {missing[:5]})" if missing else " (순서가 다릅니다)")
            )
        return X


def fit_pass_through(X_train: pd.DataFrame, pre=None, config=None) -> PassThrough:
    """학습셋의 컬럼 구성만 기억한다. 값은 보지 않으므로 누수가 생길 자리가 없다.

    `pre`와 `config`를 받고도 안 쓴다. 세 어댑터가 같은 모양이어야 `runner.prepare`가
    하나로 부를 수 있어서다. 여기서 값을 안 본다는 것 자체가 트리 쪽 명세다.
    """
    if X_train.empty:
        raise ValueError("빈 학습셋으로는 어댑터를 만들 수 없습니다.")
    return PassThrough(feature_columns=tuple(X_train.columns))
