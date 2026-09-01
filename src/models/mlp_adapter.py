"""MLP용 입력 어댑터.

공통 전처리(`preprocess.py`)는 트리를 기준으로 만들어져 있다. 글자 컬럼을 학습셋 빈도
순위 정수로 바꾸고, 결측은 그대로 두고, 크기도 안 맞춘다. 트리는 그걸로 충분하지만
MLP는 아니다.

- **MLP는 값을 크기로 읽는다.** 빈도 순위를 그대로 넣으면 card1=0과 1이 이웃이라는
  뜻이 되는데 그런 관계가 없다.
- **결측을 아예 못 먹는다.** sklearn MLPClassifier는 NaN을 보면 예외를 던진다.
- **자릿수가 컬럼마다 다르다.** id_02 최대 999,595 / TransactionAmt 31,937 / C1 4,685.
  그대로 넣으면 큰 컬럼이 학습을 지배한다.

그래서 다섯 가지를 한다. **규칙은 전부 학습셋에서만 만든다.**

1. **결측 표시** — 채우기 전에 어디가 비었는지 컬럼으로 남긴다
2. **원핫** — 값이 적은 글자 컬럼. 결측이 자기 칸을 받는다
3. **빈도 인코딩** — 값이 많은 글자 컬럼과, 숫자로 들어오는 식별자
4. **hour를 sin·cos로** — 0시와 23시가 실제로는 이웃인데 크기로 넣으면 제일 멀어진다
5. **분위수 변환** — 나머지 전부를 정규분포 모양으로 펴고, 남은 결측을 0으로 채운다

표준화 대신 분위수 변환을 쓰는 이유는 꼬리가 길어서다. V와 id_02처럼 극단값이 큰
컬럼을 표준화하면 그 몇 행이 컬럼 전체의 크기를 정해버린다. 분위수 변환은 순위만 보므로
극단값이 있어도 나머지가 눌리지 않는다.

**채우는 값이 0인 것이 곧 학습셋 중앙값이다.** 분위수 변환의 출력이 정규분포 모양이라
0이 50번째 백분위수 자리다. 변환 전에 중앙값을 따로 구해 채우는 것과 결과가 같은데,
이쪽이 더 낫다 — sklearn 분위수 변환은 fit에서 NaN을 무시하므로, 먼저 채우면 채운 값이
분포 추정에 섞여 들어간다.

**모든 컬럼이 정규분포가 되지는 않는다.** 실제 학습셋에서 재보면 TransactionAmt는 평균
-0.000·표준편차 0.998로 깔끔하게 펴지지만, V258은 -1.058·2.175이고 id_02는 0.000·0.510이다.
이유가 둘이다. 분위수 변환은 순위를 보는데 값 종류가 적으면(V258은 66종) 동점이 한 점으로
뭉쳐서 분포가 뾰족해지고, 결측률이 높으면(id_02는 병합 후 76%) 채운 0에 질량이 몰린다.
고칠 수 있는 것이 아니라 이 데이터의 성질이다. 크기가 서로 비슷한 범위에 들어오는 것이
목적이었고 그건 됐다 — 어떤 컬럼도 다른 컬럼을 자릿수로 압도하지 않는다.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from src.models.preprocess import HOUR, Preprocessor

HOURS_PER_DAY = 24


@dataclass(frozen=True)
class MlpAdapter:
    """학습셋에서 만든 변환 규칙. 만들고 나면 바뀌지 않는다."""

    # 컬럼 -> 학습셋에서 본 코드 값들. 이 코드마다 칸이 하나씩 생긴다.
    onehot: Mapping[str, tuple[int, ...]]
    # 컬럼 -> 값별 학습셋 등장 횟수
    frequency: Mapping[str, Mapping[object, int]]
    # 결측 패턴이 완전히 같은 컬럼들의 무리. 무리마다 표시 컬럼을 하나만 만든다.
    missing_groups: tuple[tuple[str, ...], ...]
    # 분위수 변환을 거는 컬럼과, 학습셋으로 맞춘 변환기
    quantile_columns: tuple[str, ...]
    quantile: QuantileTransformer
    feature_columns: tuple[str, ...]

    @property
    def missing_flags(self) -> tuple[str, ...]:
        """무리를 대표하는 컬럼 이름. 표시 컬럼 이름이 여기서 나온다."""
        return tuple(group[0] for group in self.missing_groups)

    def apply(self, X: pd.DataFrame) -> pd.DataFrame:
        """만든 규칙을 적용만 한다. 새 규칙을 만들지 않는다."""
        missing = _missing_flags(X, self.missing_groups)
        # 채우기 전에 표시를 먼저 뽑아야 한다. 채운 뒤에는 어디가 비었는지 알 수 없다.
        pieces = [missing, _onehot(X, self.onehot), _cyclic_hour(X)]

        # 분위수 변환에 들어갈 컬럼만 복사한다. 프레임 전체를 뜨면 432컬럼을 통째로
        # 한 번 더 들고 있게 된다.
        wide = X[list(self.quantile_columns)].copy()
        for column, counts in self.frequency.items():
            wide[column] = _by_frequency(X[column], counts)

        scaled = pd.DataFrame(
            self.quantile.transform(wide),
            columns=list(self.quantile_columns),
            index=X.index,
        )
        # 변환 뒤 남은 결측을 0으로 채운다. 정규분포 출력에서 0이 학습셋 중앙값 자리다.
        pieces.append(scaled.fillna(0.0))

        out = pd.concat(pieces, axis=1).astype("float32")
        if tuple(out.columns) != self.feature_columns:
            raise ValueError(
                f"만들어진 컬럼이 학습 때와 다릅니다 "
                f"(학습 {len(self.feature_columns)}개, 지금 {len(out.columns)}개)."
            )
        return out


def _onehot(X: pd.DataFrame, spec: Mapping[str, tuple[int, ...]]) -> pd.DataFrame:
    """값마다 칸을 하나씩 준다.

    학습셋에 없던 값(공통 전처리가 -1로 찍는다)은 자기 칸이 없어서 그 행이 전부 0이 된다.
    0만 늘어선 모양 자체가 '처음 보는 값'이라는 뜻이라 신호가 사라지지는 않는다.
    """
    columns = {}
    for column, codes in spec.items():
        values = X[column].to_numpy()
        for code in codes:
            columns[f"{column}={code}"] = (values == code).astype("float32")
    return pd.DataFrame(columns, index=X.index)


def _by_frequency(values: pd.Series, counts: Mapping[object, int]) -> pd.Series:
    """학습셋 등장 횟수로 바꾼다.

    원래 비어 있던 자리는 비운 채로 둔다. 나중에 분위수 변환을 거쳐 중앙값으로 채워지고,
    비었다는 사실 자체는 결측 표시 컬럼이 따로 들고 있다.

    학습셋에 없던 값은 0으로 둔다. 실제 횟수의 최솟값이 1이므로 0은 어떤 값보다도 작고,
    변환 뒤 제일 낮은 자리로 간다 — 처음 보는 카드가 드문 쪽으로 읽히는 것이 맞다.
    """
    mapped = values.map(dict(counts)).astype("float64")
    return mapped.fillna(0.0).where(values.notna(), np.nan)


def _cyclic_hour(X: pd.DataFrame) -> pd.DataFrame:
    """하루 중 시각을 원 위의 점으로 바꾼다.

    0시와 23시는 한 시간 차이인데 숫자로는 23만큼 떨어져 있다. 트리는 `hour <= 11` 처럼
    잘라 쓰면 그만이라 상관없지만, MLP는 거리를 그대로 믿어서 자정 근처가 끊긴다.
    """
    radian = 2 * np.pi * X[HOUR].to_numpy() / HOURS_PER_DAY
    return pd.DataFrame(
        {"hour_sin": np.sin(radian), "hour_cos": np.cos(radian)}, index=X.index
    )


def _missing_flags(X: pd.DataFrame, groups: tuple[tuple[str, ...], ...]) -> pd.DataFrame:
    """무리마다 '여기가 비었다'를 1로 남긴다. 이름은 무리의 첫 컬럼에서 딴다."""
    return pd.DataFrame(
        {f"{group[0]}_결측": X[group[0]].isna().astype("float32") for group in groups},
        index=X.index,
    )


def _group_by_missing_pattern(X_train: pd.DataFrame, columns) -> tuple[tuple[str, ...], ...]:
    """결측 자리가 완전히 같은 컬럼들을 한 무리로 묶는다.

    V 339개는 결측이 컬럼별로 따로 놀지 않고 블록 단위로 함께 움직인다. 하나씩 표시
    컬럼을 만들면 같은 값이 수십 개 복사되므로 무리당 하나만 만든다. 무리를 손으로 적지
    않고 결측 자리에서 찾아내므로, 나중에 다른 데이터를 붙여도 규칙이 그대로 통한다.

    학습셋에서 한 번도 안 비었거나 전부 비어 있는 컬럼은 뺀다. 표시 컬럼이 상수가 되어
    아무것도 못 알려준다.
    """
    found: dict[bytes, list[str]] = {}
    for column in columns:
        mask = X_train[column].isna().to_numpy()
        if not mask.any() or mask.all():
            continue
        key = hashlib.blake2b(mask.tobytes(), digest_size=16).digest()
        found.setdefault(key, []).append(column)
    return tuple(tuple(members) for members in found.values())


def fit_mlp_adapter(X_train: pd.DataFrame, pre: Preprocessor, config: dict) -> MlpAdapter:
    """학습셋 하나만 보고 변환 규칙을 만든다.

    `pre`가 필요한 이유는 어느 컬럼이 원래 글자였는지 알아야 해서다. 공통 전처리를 거치고
    나면 전부 정수라서 프레임만 봐서는 ProductCD(코드)와 card3(값)를 구분할 수 없다.
    """
    if X_train.empty:
        raise ValueError("빈 학습셋으로는 어댑터를 만들 수 없습니다.")

    mlp_cfg = config["mlp"]
    id_columns = [c for c in mlp_cfg["id_columns"] if c in X_train.columns]

    category = [c for c in X_train.columns if c in pre.category_columns]
    onehot_columns = [
        c for c in category if X_train[c].nunique() <= mlp_cfg["onehot_max_values"]
    ]
    frequency_columns = [c for c in category if c not in set(onehot_columns)] + id_columns

    onehot = {c: tuple(sorted(X_train[c].unique().tolist())) for c in onehot_columns}
    frequency = {
        c: dict(X_train[c].value_counts(dropna=True).items()) for c in frequency_columns
    }

    # 원핫으로 푼 컬럼은 결측이 -2 코드로 이미 칸을 받는다. 숫자 컬럼만 표시가 필요하다.
    handled = set(onehot_columns) | {HOUR}
    missing_groups = _group_by_missing_pattern(
        X_train, [c for c in X_train.columns if c not in handled]
    )

    # 빈도로 바꾼 컬럼도 크기가 제각각이라(1회부터 수십만 회까지) 함께 편다.
    quantile_columns = tuple(c for c in X_train.columns if c not in handled)

    wide = X_train[list(quantile_columns)].copy()
    for column, counts in frequency.items():
        wide[column] = _by_frequency(X_train[column], counts)

    # subsample=None이면 학습셋 전량으로 분위수를 잡는다. 일부만 뽑으면 뽑기에 따라
    # 경계가 흔들려서, 같은 설정으로 돌려도 값이 미세하게 달라진다.
    # n_quantiles를 행 수로 눌러두는 것은 작은 데이터에서 sklearn이 내는 경고를 막기
    # 위해서다. 실제 학습셋은 41만 행이라 1,000이 그대로 쓰인다.
    quantile = QuantileTransformer(
        output_distribution="normal",
        n_quantiles=min(1000, len(X_train)),
        subsample=None,
        random_state=config["seed"],
    )
    quantile.fit(wide)

    partial = MlpAdapter(
        onehot=onehot,
        frequency=frequency,
        missing_groups=missing_groups,
        quantile_columns=quantile_columns,
        quantile=quantile,
        feature_columns=(),
    )
    # 컬럼 순서는 한 행만 태워보고 확정한다. 손으로 적으면 코드와 어긋날 수 있다.
    return replace(partial, feature_columns=_build_order(partial, X_train.head(1)))


def _build_order(adapter: MlpAdapter, sample: pd.DataFrame) -> tuple[str, ...]:
    """한 행만 태워보고 나오는 컬럼 순서를 확정한다."""
    missing = _missing_flags(sample, adapter.missing_groups)
    onehot = _onehot(sample, adapter.onehot)
    hour = _cyclic_hour(sample)
    return (
        tuple(missing.columns)
        + tuple(onehot.columns)
        + tuple(hour.columns)
        + adapter.quantile_columns
    )
