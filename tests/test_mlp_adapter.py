"""MLP 입력 어댑터 검증.

이 어댑터는 `PassThrough`와 달리 **학습셋 통계를 들고 있다** — 원핫 코드 목록, 값별
등장 횟수, 분위수 경계. 그래서 누수가 실제로 생길 수 있는 첫 어댑터다. 규칙이 검증셋을
봤는지를 값을 심어놓고 확인한다.

MLP가 NaN을 아예 못 먹으므로 "결측이 하나도 안 남는다"도 여기서 고정한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.mlp_adapter import fit_mlp_adapter
from src.models.preprocess import fit_preprocessor

CONFIG = {
    "seed": 42,
    "mlp": {"onehot_max_values": 10, "id_columns": ["card1", "card2"]},
}

VAL_ONLY = "검증셋에만"


def raw_frame(n=300):
    """앞 250행이 학습, 뒤 50행이 검증. 검증 구간에만 있는 상품 값을 심어둔다.

    V1·V2는 결측 자리가 서로 같고 V3만 다르게 만들어, 무리로 묶는 규칙이 도는지 본다.
    """
    rng = np.random.default_rng(0)
    상품 = np.array(rng.choice(["W", "C", "R"], n), dtype=object)
    상품[250:] = VAL_ONLY

    m1 = np.array(rng.choice(["T", "F"], n), dtype=object)
    m1[::4] = None

    v1, v2, v3 = (rng.normal(size=n) for _ in range(3))
    v1[::3] = v2[::3] = np.nan  # 같은 결측 자리
    v3[::5] = np.nan  # 다른 결측 자리

    card2 = rng.integers(100, 600, n).astype(float)
    card2[::7] = np.nan

    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "TransactionDT": np.arange(n) * 3600 + 86_400,  # 1시간 간격
            "isFraud": (rng.random(n) < 0.2).astype(int),
            "상품": 상품,
            "M1": m1,
            "장치": np.array([f"기기{i % 40}" for i in range(n)], dtype=object),
            "card1": rng.integers(1000, 1100, n),
            "card2": card2,
            "금액": rng.gamma(2, 40, n),
            "V1": v1,
            "V2": v2,
            "V3": v3,
        }
    )


@pytest.fixture
def 준비():
    frame = raw_frame()
    train, val = frame.iloc[:250], frame.iloc[250:]
    pre = fit_preprocessor(train)
    X_train, X_val = pre.apply(train), pre.apply(val)
    return fit_mlp_adapter(X_train, pre, CONFIG), X_train, X_val


# ── 어느 컬럼을 어떻게 다루나 ────────────────────────────────────────────────


def test_값이_적은_글자는_원핫_많은_글자는_빈도로_간다(준비):
    adapter, _, _ = 준비
    assert set(adapter.onehot) == {"상품", "M1"}
    # 장치는 40종이라 원핫하면 칸이 40개 생긴다. card1·card2는 숫자지만 크기에 뜻이 없다.
    assert set(adapter.frequency) == {"장치", "card1", "card2"}


def test_원핫은_학습셋에_있던_값만_칸을_받는다(준비):
    adapter, X_train, _ = 준비
    assert adapter.onehot["상품"] == tuple(sorted(X_train["상품"].unique()))
    # M1은 T/F에 결측(-2)까지 세 칸이다. 결측이 자기 칸을 받으므로 표시 컬럼이 필요 없다.
    assert -2 in adapter.onehot["M1"]
    assert "M1_결측" not in adapter.feature_columns


def test_학습셋에_없던_값은_원핫이_전부_0이_된다(준비):
    """0만 늘어선 모양 자체가 '처음 보는 값'이라는 뜻이라 신호가 사라지지 않는다."""
    adapter, _, X_val = 준비
    칸 = [c for c in adapter.feature_columns if c.startswith("상품=")]
    assert (adapter.apply(X_val)[칸].to_numpy() == 0).all()


# ── 누수 ────────────────────────────────────────────────────────────────────


def test_규칙을_학습셋에서만_만든다(준비):
    """검증셋에만 있는 값이 규칙에 들어오면 그 순간 미래가 과거로 샌 것이다."""
    adapter, X_train, _ = 준비
    assert len(adapter.onehot["상품"]) == X_train["상품"].nunique()
    for counts in adapter.frequency.values():
        assert sum(counts.values()) <= len(X_train)


def test_빈도가_학습셋_등장_횟수와_같다(준비):
    adapter, X_train, _ = 준비
    실제 = X_train["장치"].value_counts()
    assert adapter.frequency["장치"] == dict(실제.items())


def test_처음_보는_값은_빈도_0으로_간다(준비):
    """실제 횟수는 최소 1이므로 0은 어떤 값보다 작다. 드문 쪽으로 읽히는 것이 맞다."""
    adapter, X_train, _ = 준비
    붙임 = X_train.head(1).copy()
    붙임["장치"] = 9999  # 학습셋에 없던 코드
    변환 = adapter.apply(붙임)["장치"].iloc[0]
    최소 = adapter.apply(X_train)["장치"].min()
    assert 변환 <= 최소


def test_분위수_경계를_검증셋으로_다시_맞추지_않는다(준비):
    adapter, X_train, X_val = 준비
    전 = adapter.quantile.quantiles_.copy()
    adapter.apply(X_val)
    assert np.array_equal(전, adapter.quantile.quantiles_)


# ── 결측 ────────────────────────────────────────────────────────────────────


def test_결측_자리가_같은_컬럼은_표시를_하나만_만든다(준비):
    """V1과 V2는 같은 행에서 같이 비므로 표시 컬럼 두 개는 복사본이다."""
    adapter, _, _ = 준비
    assert adapter.missing_groups == (("card2",), ("V1", "V2"), ("V3",))
    assert adapter.missing_flags == ("card2", "V1", "V3")
    assert "V2_결측" not in adapter.feature_columns


def test_한_번도_안_빈_컬럼은_표시를_안_만든다(준비):
    """표시 컬럼이 상수가 되어 아무것도 못 알려준다."""
    adapter, _, _ = 준비
    assert "금액_결측" not in adapter.feature_columns
    assert "card1_결측" not in adapter.feature_columns


def test_표시가_채우기_전_상태를_담는다(준비):
    adapter, X_train, _ = 준비
    나온것 = adapter.apply(X_train)
    assert np.array_equal(
        나온것["V1_결측"].to_numpy() == 1.0, X_train["V1"].isna().to_numpy()
    )


def test_결측이_하나도_안_남는다(준비):
    """sklearn MLPClassifier는 NaN을 보면 예외를 던진다. 검증셋도 마찬가지여야 한다."""
    adapter, X_train, X_val = 준비
    assert X_train.isna().to_numpy().any()  # 원래는 비어 있었다
    for frame in (X_train, X_val):
        assert not adapter.apply(frame).isna().to_numpy().any()


def test_채운_값이_학습셋_중앙값_자리다(준비):
    """분위수 변환 출력이 정규분포 모양이라 0이 50번째 백분위수 자리다."""
    adapter, X_train, _ = 준비
    나온것 = adapter.apply(X_train)
    assert (나온것.loc[X_train["V1"].isna(), "V1"] == 0.0).all()


# ── hour ────────────────────────────────────────────────────────────────────


def test_0시와_23시가_이웃으로_남는다(준비):
    """숫자로 넣으면 23만큼 떨어지는데 실제로는 한 시간 차이다."""
    adapter, X_train, _ = 준비
    한행 = X_train.head(1)

    def 점(시각):
        나온것 = adapter.apply(한행.assign(hour=시각))
        return 나온것[["hour_sin", "hour_cos"]].to_numpy()[0]

    거리 = lambda a, b: float(np.linalg.norm(a - b))
    assert 거리(점(23), 점(0)) < 거리(점(0), 점(6))


# ── 출력 모양 ────────────────────────────────────────────────────────────────


def test_컬럼_순서가_학습_때와_같다(준비):
    adapter, X_train, X_val = 준비
    for frame in (X_train, X_val):
        assert tuple(adapter.apply(frame).columns) == adapter.feature_columns


def test_같은_입력이면_같은_출력이_나온다(준비):
    adapter, X_train, _ = 준비
    assert np.array_equal(adapter.apply(X_train).to_numpy(), adapter.apply(X_train).to_numpy())


def test_빈_학습셋으로는_만들_수_없다():
    frame = raw_frame()
    pre = fit_preprocessor(frame.iloc[:250])
    with pytest.raises(ValueError, match="빈 학습셋"):
        fit_mlp_adapter(pre.apply(frame).head(0), pre, CONFIG)
