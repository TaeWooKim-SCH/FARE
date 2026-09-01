"""트리용 어댑터 검증.

아무것도 안 하는 층이라 볼 것이 없어 보이지만, **정말로 아무것도 안 하는지**가 이 층의
명세다. 값을 하나라도 건드리면 XGBoost와 RF의 입력이 갈라지고, 그러면 전이성 결과에서
학습 방식 차이와 입력 차이가 섞인다.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.models.adapter import PassThrough, fit_pass_through


def make_X():
    return pd.DataFrame(
        {"a": [1.0, 2.0, np.nan], "b": [-2, 0, 7], "c": [0.5, np.nan, 0.5]}
    )


def test_값을_하나도_안_바꾼다():
    X = make_X()
    out = fit_pass_through(X).apply(X)
    pd.testing.assert_frame_equal(out, X)


def test_결측을_채우지_않는다():
    """XGBoost와 RandomForest는 NaN을 직접 다룬다. 채우면 트리가 배울 신호를 지운다."""
    X = make_X()
    out = fit_pass_through(X).apply(X)
    assert out["a"].isna().sum() == 1
    assert out["c"].isna().sum() == 1


def test_학습셋의_값을_기억하지_않는다():
    """규칙이 컬럼 이름뿐이면 누수가 생길 자리가 없다. 필드가 늘면 이 검사가 깨진다."""
    names = [f.name for f in dataclasses.fields(PassThrough)]
    assert names == ["feature_columns"]


def test_컬럼_순서가_다르면_거부한다():
    """순서만 바뀌어도 모델은 예외 없이 조용히 엉뚱한 값을 읽는다."""
    X = make_X()
    adapter = fit_pass_through(X)
    with pytest.raises(ValueError, match="순서가 다릅니다"):
        adapter.apply(X[["b", "a", "c"]])


def test_컬럼이_빠지면_어느_것인지_알려준다():
    X = make_X()
    adapter = fit_pass_through(X)
    with pytest.raises(ValueError, match="빠진 컬럼: \\['b'\\]"):
        adapter.apply(X.drop(columns=["b"]))


def test_모르는_컬럼이_끼어들면_거부한다():
    X = make_X()
    adapter = fit_pass_through(X)
    with pytest.raises(ValueError, match="컬럼 구성이 학습 때와 다릅니다"):
        adapter.apply(X.assign(끼어듦=1.0))


def test_빈_학습셋으로는_만들_수_없다():
    with pytest.raises(ValueError, match="빈 학습셋"):
        fit_pass_through(pd.DataFrame())
