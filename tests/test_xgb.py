"""학습 코드 검증.

성능이 잘 나오는지가 아니라, **평가셋이 학습에 닿지 않는지**와 **설정이 기록되는지**를 본다.
성능은 실제 데이터로만 의미가 있어서 여기서 재지 않는다.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.xgb import feature_importance, train_xgb

CONFIG = {
    "seed": 42,
    "xgboost": {
        "n_estimators": 20,
        "learning_rate": 0.3,
        "max_depth": 3,
        "early_stopping_rounds": 5,
        "eval_metric": "aucpr",
    },
}


def make_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.integers(0, 5, n)}
    )
    y = pd.Series((X["a"] + rng.normal(0, 0.3, n) > 0.8).astype(int))
    return X, y


def test_학습이_되고_점수가_확률로_나온다():
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    s = t.score(X[300:])
    assert len(s) == 100
    assert ((s >= 0) & (s <= 1)).all()


def test_평가셋을_넘길_자리가_없다():
    """인자가 학습셋과 검증셋뿐이라, 평가셋을 넣으려면 함수를 고쳐야 한다."""
    import inspect

    params = list(inspect.signature(train_xgb).parameters)
    assert params == ["X_train", "y_train", "X_val", "y_val", "config"]


def test_설정과_행_수를_기록한다():
    """결과 파일에 같이 적어야 나중에 같은 숫자를 다시 만들 수 있다."""
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    assert t.params["random_state"] == 42
    assert t.params["learning_rate"] == 0.3
    assert t.train_rows == 300
    assert t.val_rows == 100
    assert t.feature_names == ("a", "b", "c")


def test_같은_시드면_같은_점수가_나온다():
    X, y = make_xy()
    a = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG).score(X[300:])
    b = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG).score(X[300:])
    assert np.array_equal(a, b)


def test_검증셋에서_멈출_시점을_정한다():
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    assert 0 <= t.best_iteration < CONFIG["xgboost"]["n_estimators"]


def test_학습셋과_검증셋의_컬럼이_다르면_거부한다():
    X, y = make_xy()
    with pytest.raises(ValueError, match="컬럼이 다릅니다"):
        train_xgb(X[:300], y[:300], X[300:].drop(columns=["b"]), y[300:], CONFIG)


def test_학습셋에_사기가_없으면_거부한다():
    """전부 정상이면 학습이 되긴 하는데 아무 뜻이 없는 모델이 나온다."""
    X, y = make_xy()
    with pytest.raises(ValueError, match="사기 거래가 없"):
        train_xgb(X[:300], y[:300] * 0, X[300:], y[300:], CONFIG)


def test_점수를_낼_때_컬럼_순서가_다르면_거부한다():
    """순서만 바뀌어도 모델은 조용히 엉뚱한 값을 읽는다."""
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    with pytest.raises(ValueError, match="컬럼 구성이 학습 때와 다릅니다"):
        t.score(X[300:][["b", "a", "c"]])


def test_분기_횟수와_총_기여를_함께_돌려준다():
    """gain 하나만 보면 한 번 쓰고 크게 갈라진 컬럼이 1등이 된다. 둘 다 봐야 한다."""
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    imp = feature_importance(t, top=3)
    assert list(imp.columns) == [
        "feature", "splits", "total_gain", "splits_share", "total_gain_share"
    ]
    assert imp["total_gain_share"].sum() == pytest.approx(1.0)
    # a로만 정답을 만들었으니 a가 1등이어야 한다
    assert imp.iloc[0]["feature"] == "a"


def test_안_쓴_컬럼도_0으로_남는다():
    """빠뜨리면 컬럼 수가 달라져 나중에 모델끼리 비교할 때 어긋난다."""
    X, y = make_xy()
    X["안쓰임"] = 0.0
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    imp = feature_importance(t, top=99)
    row = imp[imp["feature"] == "안쓰임"]
    assert len(row) == 1
    assert row.iloc[0]["splits"] == 0
