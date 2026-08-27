"""전처리 검증.

여기서 학습셋 밖의 정보가 새면 이후 모든 성능 수치가 부풀려진다. 규칙을 학습셋에서만
만드는지, 처음 보는 값을 어떻게 다루는지, 그리고 다시 돌려도 같은 숫자가 나오는지를 고정한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.preprocess import (
    HOUR,
    LABEL,
    MISSING_CODE,
    TIME,
    UNSEEN_CODE,
    add_hour,
    fit_preprocessor,
    target_of,
)


def make_frame(products, amounts=None, fraud=None, hours=None) -> pd.DataFrame:
    n = len(products)
    hours = hours if hours is not None else [0] * n
    return pd.DataFrame(
        {
            "TransactionID": range(n),
            TIME: [86400 + h * 3600 for h in hours],
            LABEL: fraud if fraud is not None else [0] * n,
            "TransactionAmt": amounts if amounts is not None else [10.0] * n,
            "ProductCD": products,
        }
    )


# ── 글자를 숫자로 ──────────────────────────────────────────────────────────


def test_글자_값이_숫자로_바뀐다():
    train = make_frame(["W", "W", "C"])
    out = fit_preprocessor(train).apply(train)
    assert out["ProductCD"].tolist() == [0, 0, 1]


def test_자주_나오는_값이_작은_번호를_받는다():
    pre = fit_preprocessor(make_frame(["C", "W", "W", "W", "C"]))
    assert pre.codes["ProductCD"]["W"] == 0
    assert pre.codes["ProductCD"]["C"] == 1


def test_빈도가_같으면_값_자체로_순서를_정한다():
    """행 순서가 번호를 정하면, 데이터를 섞어 돌릴 때마다 숫자의 뜻이 달라진다."""
    pre = fit_preprocessor(make_frame(["C", "W"]))
    assert pre.codes["ProductCD"] == {"C": 0, "W": 1}


def test_행_순서를_섞어도_같은_번호가_나온다():
    values = ["W", "C", "H", "S", "R", "W", "C", "H"]
    first = fit_preprocessor(make_frame(values)).codes["ProductCD"]
    shuffled = fit_preprocessor(make_frame(list(reversed(values)))).codes["ProductCD"]
    assert dict(first) == dict(shuffled)


# ── 처음 보는 값과 빈 값 ───────────────────────────────────────────────────


def test_학습셋에_없던_값은_따로_표시된다():
    pre = fit_preprocessor(make_frame(["W", "W", "C"]))
    out = pre.apply(make_frame(["W", "H"]))
    assert out["ProductCD"].tolist() == [0, UNSEEN_CODE]


def test_빈_값은_없던_값과_다른_번호를_받는다():
    pre = fit_preprocessor(make_frame(["W", "W", None]))
    out = pre.apply(make_frame(["W", None, "H"]))
    assert out["ProductCD"].tolist() == [0, MISSING_CODE, UNSEEN_CODE]


def test_category형이어도_학습셋에_없던_값은_없던_값이다():
    """category형은 학습셋에 안 나온 값도 개수 0으로 딸려 나온다. 그대로 두면 정상 번호를 받는다."""
    train = make_frame(["W", "W", "C"])
    train["ProductCD"] = pd.Categorical(train["ProductCD"], categories=["W", "C", "H", "S"])
    pre = fit_preprocessor(train)
    assert "H" not in pre.codes["ProductCD"]

    later = make_frame(["H"])
    later["ProductCD"] = pd.Categorical(later["ProductCD"], categories=["W", "C", "H", "S"])
    assert pre.apply(later)["ProductCD"].tolist() == [UNSEEN_CODE]


# ── 학습셋 밖 정보가 새지 않는가 ───────────────────────────────────────────


def test_검증셋을_넣어도_규칙이_바뀌지_않는다():
    pre = fit_preprocessor(make_frame(["W", "W", "C"]))
    before = dict(pre.codes["ProductCD"])
    pre.apply(make_frame(["H", "S", "R"]))
    assert dict(pre.codes["ProductCD"]) == before


def test_규칙을_밖에서_고칠_수_없다():
    """frozen은 통째로 바꾸는 것만 막는다. 안쪽 값까지 잠가야 한다."""
    pre = fit_preprocessor(make_frame(["W", "C"]))
    with pytest.raises(TypeError):
        pre.codes["ProductCD"]["H"] = 99
    with pytest.raises(TypeError):
        pre.codes["새컬럼"] = {}
    with pytest.raises(Exception):
        pre.feature_columns = ("아무거나",)


def test_규칙을_만든_학습셋_구간을_남긴다():
    """결과 파일에 함께 적어 '학습셋으로만 만들었다'를 숫자로 보이기 위해서다."""
    pre = fit_preprocessor(make_frame(["W", "C", "H"], hours=[1, 5, 9]))
    assert pre.fit_rows == 3
    assert pre.fit_time_range == (86400 + 3600, 86400 + 9 * 3600)


# ── 빼는 컬럼과 만드는 컬럼 ────────────────────────────────────────────────


def test_거래_번호와_원본_시각은_모델에_넣지_않는다():
    pre = fit_preprocessor(make_frame(["W", "C"]))
    assert "TransactionID" not in pre.feature_columns
    assert TIME not in pre.feature_columns


def test_하루_중_시각을_만들어_넣는다():
    pre = fit_preprocessor(make_frame(["W", "C", "H"], hours=[0, 13, 23]))
    assert HOUR in pre.feature_columns
    assert pre.apply(make_frame(["W", "C", "H"], hours=[0, 13, 23]))[HOUR].tolist() == [0, 13, 23]


def test_시각_파생은_하루를_넘어가면_돌아온다():
    frame = pd.DataFrame({TIME: [0, 3600 * 25, 3600 * 47]})
    assert add_hour(frame)[HOUR].tolist() == [0, 1, 23]


def test_정답은_입력에서_빠진다():
    assert LABEL not in fit_preprocessor(make_frame(["W", "C"])).feature_columns


def test_숫자_컬럼은_그대로_둔다():
    train = make_frame(["W", "C"], amounts=[10.5, 20.25])
    assert fit_preprocessor(train).apply(train)["TransactionAmt"].tolist() == [10.5, 20.25]


def test_빈_값을_채우지_않는다():
    train = pd.DataFrame(
        {LABEL: [0, 0], "TransactionID": [1, 2], TIME: [86400, 90000], "C1": [1.0, np.nan]}
    )
    assert fit_preprocessor(train).apply(train)["C1"].isna().sum() == 1


# ── 잘못된 입력은 거부 ─────────────────────────────────────────────────────


def test_학습_때와_형이_다르면_거부한다():
    """글자였던 컬럼이 숫자로 오면 전부 '처음 보는 값'이 되어 조용히 망가진다."""
    pre = fit_preprocessor(make_frame(["W", "C"]))
    later = make_frame(["W", "C"])
    later["ProductCD"] = [1, 2]
    with pytest.raises(TypeError, match="형이 학습 때와 다릅니다"):
        pre.apply(later)


def test_시각형_컬럼은_거부한다():
    train = make_frame(["W", "C"])
    train["언제"] = pd.to_datetime(["2018-01-01", "2018-01-02"])
    with pytest.raises(TypeError, match="시각형"):
        fit_preprocessor(train)


def test_정답이_없으면_거부한다():
    with pytest.raises(KeyError, match=LABEL):
        fit_preprocessor(pd.DataFrame({"TransactionID": [1], TIME: [1], "ProductCD": ["W"]}))


def test_빈_학습셋은_거부한다():
    with pytest.raises(ValueError, match="빈 학습셋"):
        fit_preprocessor(make_frame([]))


def test_학습셋에_있던_컬럼이_없으면_거부한다():
    pre = fit_preprocessor(make_frame(["W", "C"]))
    with pytest.raises(KeyError, match="컬럼이 없습니다"):
        pre.apply(pd.DataFrame({"TransactionAmt": [1.0], TIME: [86400]}))


def test_정답만_따로_꺼낼_수_있다():
    assert target_of(make_frame(["W", "C"], fraud=[0, 1])).tolist() == [0, 1]
