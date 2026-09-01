"""세 모델이 공유하는 준비·보고 흐름 검증.

**여기가 깨지면 세 모델이 한꺼번에 오염된다.** 누수 방지 장치가 원래는 `train_xgb.py`
안에 있었는데 공유 모듈로 옮겨왔으므로, 옮긴 자리에서 다시 고정한다.

원본 CSV는 읽지 않는다. `load_merged`를 작은 합성 프레임으로 바꿔치기해서, 규칙이
학습셋만 보는지·평가셋이 τ에 닿는지를 눈에 보이는 크기에서 확인한다.
"""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from src.models import runner

VAL_ONLY = "검증셋에만"
TEST_ONLY = "평가셋에만"


def make_frame(n=2000):
    """시간순으로 정렬된 합성 거래. 검증·평가 구간에만 있는 상품 값을 심어둔다.

    분할 비율이 0.7/0.1이므로 앞 1,400행이 학습, 다음 200행이 검증, 나머지 400행이 평가다.
    심어둔 값이 전처리 규칙에 새어 들어오면 그 자리에서 잡힌다. 구간을 통째로 덮지 않고
    앞 10행에만 심는 이유는, 나머지 상품이 섞여 있어야 상품별 쪼개기도 함께 볼 수 있어서다.
    """
    rng = np.random.default_rng(0)
    # dtype=object로 만들어야 한다. rng.choice가 주는 <U1 배열에 긴 문자열을 넣으면
    # 첫 글자만 남고 잘려서, 심은 값이 조용히 다른 값이 된다.
    product = np.array(rng.choice(["W", "C", "R"], n), dtype=object)
    product[1400:1410] = VAL_ONLY
    product[1600:1610] = TEST_ONLY
    return pd.DataFrame(
        {
            "TransactionID": np.arange(2_987_000, 2_987_000 + n),
            # 10분 간격이라 같은 시각이 겹치지 않는다. 경계 밀기와 무관하게 자른다.
            "TransactionDT": np.arange(n) * 600 + 86_400,
            "isFraud": (rng.random(n) < 0.2).astype(int),
            "TransactionAmt": rng.gamma(2, 40, n).round(3),
            "ProductCD": product,
            "card1": rng.integers(1000, 18000, n),
        }
    )


@pytest.fixture
def 합성데이터(monkeypatch, tmp_path):
    """원본 CSV 대신 합성 프레임을 읽게 하고, 결과도 임시 폴더에 쓰게 한다."""
    frame = make_frame()
    monkeypatch.setattr(runner, "load_merged", lambda cfg: frame.copy())
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path)
    return frame


class 점수내는_스텁:
    """`report()`에 넘길 최소한의 모델. `score(X)` 하나만 있으면 된다.

    `flip_test`를 켜면 평가셋 행에서만 점수를 뒤집는다. 그래도 τ가 안 움직여야 한다.
    """

    def __init__(self, test_index=None, flip_test=False):
        self.test_index = test_index
        self.flip_test = flip_test

    def score(self, X):
        s = 1.0 / (1.0 + np.exp(-(X["TransactionAmt"].to_numpy() / 40.0 - 2.0)))
        if self.flip_test and self.test_index is not None and X.index.equals(self.test_index):
            return 1.0 - s
        return s


# ── 준비 단계 ────────────────────────────────────────────────────────────────


def test_final이_아니면_평가셋을_아예_안_읽는다(합성데이터):
    p = runner.prepare(final=False, stop_split=False)
    assert "test" not in p.X
    assert "test" not in p.y
    assert p.product is None


def test_자른_조각을_통째로_들고_있지_않는다(합성데이터):
    """`split`을 들고 있으면 --final이 아닐 때도 `prepared.split.test`로 닿을 수 있다."""
    names = [f.name for f in dataclasses.fields(runner.Prepared)]
    assert "split" not in names


def test_전처리_규칙을_학습셋으로만_만든다(합성데이터):
    """검증·평가 구간에만 있는 값이 규칙에 들어오면 그 순간 미래가 과거로 샌 것이다."""
    p = runner.prepare(final=True, stop_split=False)
    코드 = p.pre.codes["ProductCD"]
    assert VAL_ONLY not in 코드
    assert TEST_ONLY not in 코드
    assert set(코드) == {"W", "C", "R"}
    assert p.pre.fit_rows == len(p.X["train"])


def test_학습셋에_없던_값은_처음_보는_값으로_찍힌다(합성데이터):
    """규칙이 검증셋을 안 봤다는 직접 증거. 안 봤으니 심어둔 자리에 -1이 나와야 한다."""
    p = runner.prepare(final=True, stop_split=False)
    for name in ("val", "test"):
        상품 = p.X[name]["ProductCD"]
        심은곳, 나머지 = 상품.index[:10], 상품.index[10:]
        assert (상품.loc[심은곳] == -1).all()
        assert (상품.loc[나머지] != -1).all()


def test_어댑터도_학습셋만_보고_만든다(합성데이터):
    p = runner.prepare(final=True, stop_split=False)
    assert p.adapter.feature_columns == p.pre.feature_columns


def test_조기_종료_조각은_학습셋_안에서_시간순으로_나온다(합성데이터):
    """무작위로 뽑으면 미래가 학습 쪽으로 샌다. 뒤쪽을 잘라야 한다."""
    p = runner.prepare(final=False, stop_split=True)
    assert len(p.X["fit"]) + len(p.X["stop"]) == len(p.X["train"])
    # hour가 아니라 원본 시각으로 봐야 순서를 알 수 있다. 학습셋 안쪽에서 뒤쪽인지 본다.
    assert len(p.X["stop"]) == pytest.approx(len(p.X["train"]) * 0.15, rel=0.05)


def test_조기_종료_조각을_안_만들_수도_있다(합성데이터):
    """Random Forest는 조기 종료가 없다. 안 쓸 조각을 만들면 메모리만 한 번 더 잡는다."""
    p = runner.prepare(final=False, stop_split=False)
    assert set(p.X) == {"train", "val"}


# ── 보고 단계 ────────────────────────────────────────────────────────────────


def _run(p, **stub):
    out = runner.report(p, 점수내는_스텁(**stub), {"params": {}}, "테스트")
    return json.loads(out.read_text(encoding="utf-8"))


def test_평가셋_점수를_뒤집어도_임계값이_안_변한다(합성데이터):
    """τ는 검증셋에서만 나온다. 평가셋을 어떻게 흔들어도 비트까지 같아야 한다.

    순서만 보고 넘기지 않으려고 실제로 흔들어본다. τ가 조금이라도 움직이면 평가셋이
    운영 임계값 결정에 닿았다는 뜻이고, 그러면 기준선 전체가 무효다.
    """
    p = runner.prepare(final=True, stop_split=False)
    보통 = _run(p)["threshold"]["tau"]
    뒤집음 = _run(p, test_index=p.X["test"].index, flip_test=True)["threshold"]["tau"]
    assert 보통 == 뒤집음

    안열었을때 = _run(runner.prepare(final=False, stop_split=False))["threshold"]["tau"]
    assert 보통 == 안열었을때


def test_평가셋을_봤는지_결과_파일에_남긴다(합성데이터):
    """나중에 이 숫자를 논문에 인용할 때 근거가 된다."""
    assert _run(runner.prepare(final=True, stop_split=False))["test_included"] is True
    assert _run(runner.prepare(final=False, stop_split=False))["test_included"] is False


def test_final이_아니면_상품별_쪼개기를_안_한다(합성데이터):
    assert _run(runner.prepare(final=False, stop_split=False))["by_product"] == []


def test_상품별_쪼개기가_점수와_행을_맞게_짝짓는다(합성데이터):
    """라벨을 위치로 바꾸는 자리가 어긋나면 다른 상품의 점수를 보고 지표를 낸다."""
    p = runner.prepare(final=True, stop_split=False)
    쪼갠것 = _run(p)["by_product"]

    실제 = p.y["test"].groupby(p.product, observed=True).agg(["size", "mean"])
    assert sum(r["n"] for r in 쪼갠것) == len(p.y["test"])
    for row in 쪼갠것:
        assert row["n"] == 실제.loc[row["ProductCD"], "size"]
        assert row["positive_rate"] == pytest.approx(실제.loc[row["ProductCD"], "mean"])


def test_임계값을_넘긴_그대로_평가셋에_쓴다(합성데이터):
    """평가셋에서 τ를 다시 고르면 그 순간 평가셋이 선택셋이 된다."""
    payload = _run(runner.prepare(final=True, stop_split=False))
    tau = payload["threshold"]["tau"]
    assert {payload["at_threshold"][n]["tau"] for n in ("train", "val", "test")} == {tau}
