"""Random Forest 학습 코드 검증.

성능이 잘 나오는지가 아니라, **평가셋이 학습에 닿지 않는지**와 **설정이 기록되는지**를 본다.
성능은 실제 데이터로만 의미가 있어서 여기서 재지 않는다.
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.rf import feature_importance, train_rf

CONFIG = {
    "seed": 42,
    "random_forest": {
        "n_estimators": 10,
        "max_depth": 4,
        "min_samples_leaf": 5,
        "max_features": "sqrt",
        "class_weight": None,
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
    s = train_rf(X, y, CONFIG).score(X)
    assert len(s) == len(X)
    assert ((s >= 0) & (s <= 1)).all()


def test_결측이_있어도_그대로_학습된다():
    """sklearn 1.4부터 숲이 NaN을 직접 다룬다. 채우면 XGBoost와 입력이 갈라진다."""
    X, y = make_xy()
    X.loc[::5, "b"] = np.nan
    s = train_rf(X, y, CONFIG).score(X)
    assert not np.isnan(s).any()


def test_검증셋과_평가셋을_넘길_자리가_없다():
    """인자가 학습셋 하나뿐이라, 검증셋을 넣으려면 함수를 고쳐야 한다."""
    import inspect

    assert list(inspect.signature(train_rf).parameters) == ["X_train", "y_train", "config"]


def test_설정과_행_수를_기록한다():
    """결과 파일에 같이 적어야 나중에 같은 숫자를 다시 만들 수 있다."""
    X, y = make_xy()
    t = train_rf(X, y, CONFIG)
    assert t.params["random_state"] == 42
    assert t.params["n_estimators"] == 10
    assert t.params["max_features"] == "sqrt"
    assert t.fit_rows == len(X)
    assert t.feature_names == ("a", "b", "c")


def test_불균형을_가중치로_밀지_않는다():
    """가중치로 한 번 밀고 τ로 또 미는 것은 같은 조정을 두 번 하는 셈이다."""
    X, y = make_xy()
    assert train_rf(X, y, CONFIG).params["class_weight"] is None


def test_같은_시드면_같은_점수가_나온다():
    X, y = make_xy()
    a = train_rf(X, y, CONFIG).score(X)
    b = train_rf(X, y, CONFIG).score(X)
    assert np.array_equal(a, b)


def test_같은_모델을_여러_번_불러도_비트까지_같다():
    """예측을 병렬로 하면 나무별 결과를 더하는 순서가 스레드마다 달라져 마지막 비트가
    흔들린다. τ 경계에 걸친 거래는 그 정도로도 판정이 뒤집혀서 회피율이 안 맞는다.
    """
    X, y = make_xy()
    t = train_rf(X, y, CONFIG)
    first = t.score(X)
    assert all(np.array_equal(first, t.score(X)) for _ in range(3))


def test_예측은_한_스레드로_고정된다():
    """위 흔들림을 없애는 장치. 학습은 병렬로 하되 예측만 1로 못 박는다."""
    X, y = make_xy()
    t = train_rf(X, y, CONFIG)
    assert t.model.n_jobs == 1
    assert t.params["n_jobs"] == 8  # 기록에는 학습에 쓴 값이 남는다


def test_config에_적은_만큼_나무를_기른다():
    X, y = make_xy()
    t = train_rf(X, y, CONFIG)
    assert len(t.model.estimators_) == CONFIG["random_forest"]["n_estimators"]


def test_학습셋에_사기가_없으면_거부한다():
    """전부 정상이면 학습이 되긴 하는데 아무 뜻이 없는 모델이 나온다."""
    X, y = make_xy()
    with pytest.raises(ValueError, match="학습셋에 사기 거래가 없"):
        train_rf(X, y * 0, CONFIG)


def test_점수를_낼_때_컬럼_순서가_다르면_거부한다():
    """순서만 바뀌어도 모델은 조용히 엉뚱한 값을 읽는다."""
    X, y = make_xy()
    t = train_rf(X, y, CONFIG)
    with pytest.raises(ValueError, match="컬럼 구성이 학습 때와 다릅니다"):
        t.score(X[["b", "a", "c"]])


def test_저장한_파일을_불러도_같은_점수가_나온다():
    """공격 단계에서 이 파일을 불러다 쓰므로, 어긋나면 회피율이 미묘하게 안 맞는다."""
    X, y = make_xy()
    t = train_rf(X, y, CONFIG)
    expected = t.score(X)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "forest.joblib"
        size = t.save(path)
        assert size == path.stat().st_size > 0
        got = joblib.load(path).predict_proba(X)[:, 1]

    assert np.array_equal(expected, got)


def test_중요도를_컬럼_전부에_대해_돌려준다():
    """안 쓴 컬럼이 빠지면 컬럼 수가 달라져 나중에 모델끼리 비교할 때 어긋난다."""
    X, y = make_xy()
    X["안쓰임"] = 0.0
    imp = feature_importance(train_rf(X, y, CONFIG), top=99)
    assert list(imp.columns) == ["feature", "mdi"]
    assert set(imp["feature"]) == {"a", "b", "c", "안쓰임"}
    assert imp.iloc[0]["feature"] == "a"  # a로만 정답을 만들었다
    assert imp[imp["feature"] == "안쓰임"].iloc[0]["mdi"] == 0.0
