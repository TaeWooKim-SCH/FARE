"""전처리 — 글자로 된 값을 숫자로 바꾼다.

규칙은 학습셋으로만 만들고 검증셋·평가셋에는 적용만 한다.

**이 함수가 학습셋만 받는다고 해서 실수가 막히지는 않는다.** `pd.concat`으로 합쳐 넘기면
그만이다. 인자를 하나로 좁힌 것은 합치려면 한 줄을 더 쓰게 만들어 눈에 띄게 하려는 것이지
보장이 아니다. 그래서 규칙을 만든 학습셋이 어떤 구간이었는지를 함께 남기고, 결과 파일에
같이 적어 "학습셋으로만 만들었다"를 숫자로 보일 수 있게 한다.

빈 값은 여기서 채우지 않는다. XGBoost는 빈 값을 그대로 다루지만 Random Forest와 MLP는
못 하므로, 채우는 방식은 모델마다 다르다. 모델별 처리는 각 모델 코드에서 한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import pandas as pd
from pandas.api.types import is_numeric_dtype

LABEL = "isFraud"
TIME = "TransactionDT"

# 모델에 넣지 않는 컬럼. 둘을 함께 빼는 이유가 다르다.
#
# - TransactionDT: 시간순으로 잘랐으니 평가셋 값이 전부 학습셋보다 크다(학습 최대
#   10,437,996 / 평가 최소 12,192,900, 겹치는 행 0%). 트리가 고르는 기준값은 학습셋
#   안에서만 나오므로 평가셋 118,108행이 통째로 한쪽으로 떨어진다. 실제로 재보면
#   기준값 2,386,656이 뽑히고 평가셋은 100%가 오른쪽이다 — 모든 행에 같은 값을 주는 셈이다.
#   크기를 다시 맞춰도 소용없다. min-max·표준화는 물론이고 규칙을 어기고 평가셋까지 보고
#   맞춰도 여전히 100%가 한쪽이다. 스케일링은 순서를 안 바꾸는데 문제가 순서이기 때문이다.
#   더 중요한 건 공격 쪽이다. 1차 공격이 시각을 옮기는데 DT가 입력이면, 회피가 시간대
#   때문인지 학습에서 못 본 구간으로 나가서인지 못 가른다. 대신 하루 중 시각을 뽑아 쓴다.
#
# - TransactionID: 시각을 순위로 바꾼 값이다. DT 순으로 정렬하면 ID도 오름차순이고 순위
#   상관이 1.000이라, DT만 빼고 ID를 남기면 아무것도 안 한 것이 된다. 게다가 공격자가
#   바꿀 수 없는 값이라 모델이 여기 기대면 금액·시각을 아무리 흔들어도 점수가 안 움직인다.
#   그러면 "공격이 안 통했다"는 결과가 나오고 그걸 방어력으로 오해한다(feature-taxonomy.md 3절).
DROPPED = ("TransactionID", TIME)

# 빈 값과 "학습셋에 없던 값"은 다른 번호를 준다. 빈 값 자체가 신호이기 때문이다
# — M1-M9는 값보다 비어 있는지가 사기율을 가른다.
MISSING_CODE = -2
UNSEEN_CODE = -1

HOUR = "hour"


def add_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """하루 중 시각(0~23)을 만든다.

    시간대에 따라 사기율이 흔들린다 — 학습셋에서 13시 2.17%부터 8시 10.01%까지 4.62배다.

    DT를 뺐는데 여기서 시각을 다시 넣는 것이 앞뒤가 안 맞아 보이지만, 24마다 되돌아온다는
    점이 다르다. 학습셋도 평가셋도 값이 0~23이라 학습에서 만든 기준이 평가셋에서도 갈린다.
    실제로 `hour <= 11`로 잘라보면 평가셋이 26.5% 대 73.5%로 나뉘고, 학습이 배운 사기율
    4.30%/3.20%가 평가셋 실제 4.29%/3.13%와 거의 맞는다. D1~D15가 카드마다 0에서 다시
    시작하는 것도 같은 이치다.
    """
    if TIME not in frame.columns:
        raise KeyError(f"{TIME}이 있어야 시각을 뽑을 수 있습니다.")
    return frame.assign(**{HOUR: ((frame[TIME] // 3600) % 24).astype("int16")})


@dataclass(frozen=True)
class Preprocessor:
    """학습셋에서 만든 변환 규칙. 만들고 나면 바뀌지 않는다."""

    codes: Mapping[str, Mapping[object, int]]
    feature_columns: tuple[str, ...]
    category_columns: frozenset[str]
    fit_rows: int
    fit_time_range: tuple[int, int]

    def apply(self, frame: pd.DataFrame) -> pd.DataFrame:
        """규칙을 그대로 적용한다. 새 규칙을 만들지 않는다."""
        frame = add_hour(frame) if HOUR in self.feature_columns and HOUR not in frame else frame

        absent = [c for c in self.feature_columns if c not in frame.columns]
        if absent:
            raise KeyError(f"학습셋에 있던 컬럼이 없습니다: {absent[:5]}")

        # 학습 때 글자였던 컬럼이 지금 숫자로 오면(또는 반대면) 규칙이 통째로 어긋난다.
        # 조용히 전부 '처음 보는 값'이 되므로 예외를 던져 알린다.
        for column in self.feature_columns:
            was_category = column in self.category_columns
            is_category = not is_numeric_dtype(frame[column])
            if was_category != is_category:
                raise TypeError(
                    f"{column}의 형이 학습 때와 다릅니다"
                    f"(학습 때 {'글자' if was_category else '숫자'}, 지금 {'글자' if is_category else '숫자'})."
                )

        out = frame[list(self.feature_columns)].copy()
        for column, mapping in self.codes.items():
            values = out[column]
            encoded = values.map(dict(mapping))
            encoded = encoded.where(values.notna(), MISSING_CODE)
            encoded = encoded.where(values.isna() | encoded.notna(), UNSEEN_CODE)
            out[column] = encoded.astype("int32")
        return out


def _category_columns(frame: pd.DataFrame) -> list[str]:
    # datetime을 글자로 보면 타임스탬프 하나하나가 번호를 받는다. 숫자도 글자도 아닌 것은 거부한다.
    columns = []
    for c in frame.columns:
        kind = frame[c].dtype.kind
        if kind in "Mm":
            raise TypeError(f"{c}가 시각형입니다. 숫자로 바꾼 뒤 넣으세요.")
        if not is_numeric_dtype(frame[c]):
            columns.append(c)
    return columns


def fit_preprocessor(train: pd.DataFrame) -> Preprocessor:
    """학습셋 하나만 받아 변환 규칙을 만든다."""
    if LABEL not in train.columns:
        raise KeyError(f"{LABEL}이 있어야 합니다. 학습셋을 넘겼는지 확인하세요.")
    if train.empty:
        raise ValueError("빈 학습셋으로는 규칙을 만들 수 없습니다.")

    time_range = (int(train[TIME].min()), int(train[TIME].max())) if TIME in train else (0, 0)
    train = add_hour(train) if TIME in train.columns else train

    features = [c for c in train.columns if c != LABEL and c not in DROPPED]
    category_columns = _category_columns(train[features])

    codes: dict[str, Mapping[object, int]] = {}
    for column in category_columns:
        counts = train[column].value_counts(dropna=True)
        # category형이면 학습셋에 없던 값도 개수 0으로 딸려 나온다. 그대로 두면
        # 평가셋에만 있는 값이 정상 번호를 받아 '처음 보는 값'으로 안 잡힌다.
        counts = counts[counts > 0]
        # 빈도가 같을 때 등장 순서로 번호를 매기면 행 순서만 바뀌어도 숫자의 뜻이 달라진다.
        # 실제로 DeviceInfo는 값의 96.6%가 동점이다. 값 자체로 동점을 깨서 고정한다.
        ordered = sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))
        codes[column] = MappingProxyType({value: number for number, (value, _) in enumerate(ordered)})

    return Preprocessor(
        codes=MappingProxyType(codes),
        feature_columns=tuple(features),
        category_columns=frozenset(category_columns),
        fit_rows=len(train),
        fit_time_range=time_range,
    )


def target_of(frame: pd.DataFrame) -> pd.Series:
    """정답만 따로 꺼낸다. 입력과 정답을 같이 넘기다 섞는 일을 막는다."""
    if LABEL not in frame.columns:
        raise KeyError(f"{LABEL}이 없습니다.")
    return frame[LABEL]
