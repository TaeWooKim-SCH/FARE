"""학습 코드 검증.

성능이 잘 나오는지가 아니라, **평가셋이 학습에 닿지 않는지**와 **설정이 기록되는지**를 본다.
성능은 실제 데이터로만 의미가 있어서 여기서 재지 않는다.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.xgb import feature_importance, refit_on_all, train_xgb

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


def test_검증셋과_평가셋을_넘길_자리가_없다():
    """인자가 학습셋 두 조각뿐이라, 검증셋이나 평가셋을 넣으려면 함수를 고쳐야 한다."""
    import inspect

    params = list(inspect.signature(train_xgb).parameters)
    assert params == ["X_fit", "y_fit", "X_stop", "y_stop", "config"]


def test_설정과_행_수를_기록한다():
    """결과 파일에 같이 적어야 나중에 같은 숫자를 다시 만들 수 있다."""
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    assert t.params["random_state"] == 42
    assert t.params["learning_rate"] == 0.3
    assert t.fit_rows == 300
    assert t.stop_rows == 100
    assert t.feature_names == ("a", "b", "c")


def test_같은_시드면_같은_점수가_나온다():
    X, y = make_xy()
    a = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG).score(X[300:])
    b = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG).score(X[300:])
    assert np.array_equal(a, b)


def test_뒤쪽_조각에서_멈출_시점을_정한다():
    X, y = make_xy()
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    assert 0 <= t.best_iteration < CONFIG["xgboost"]["n_estimators"]


def test_두_조각의_컬럼이_다르면_거부한다():
    X, y = make_xy()
    with pytest.raises(ValueError, match="컬럼이 다릅니다"):
        train_xgb(X[:300], y[:300], X[300:].drop(columns=["b"]), y[300:], CONFIG)


def test_학습_조각에_사기가_없으면_거부한다():
    """전부 정상이면 학습이 되긴 하는데 아무 뜻이 없는 모델이 나온다."""
    X, y = make_xy()
    with pytest.raises(ValueError, match="학습 조각에 사기 거래가 없"):
        train_xgb(X[:300], y[:300] * 0, X[300:], y[300:], CONFIG)


def test_조기_종료용_조각에_사기가_없으면_거부한다():
    """멈출 시점을 정할 근거가 없으면 조기 종료가 아무 데서나 걸린다."""
    X, y = make_xy()
    with pytest.raises(ValueError, match="조기 종료용 조각에 사기 거래가 없"):
        train_xgb(X[:300], y[:300], X[300:], y[300:] * 0, CONFIG)


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


def test_저장한_파일을_그냥_불러도_같은_점수가_나온다():
    """조기 종료가 남긴 안 쓸 나무를 안 자르면, 같은 파일인데 부르는 방법에 따라 점수가 달라진다.

    공격 단계에서 이 파일을 불러다 쓰므로, 어긋나면 회피율이 미묘하게 안 맞는다.
    """
    import xgboost as xgb

    X, y = make_xy(n=1200, seed=3)
    t = train_xgb(X[:900], y[:900], X[900:], y[900:], CONFIG)
    expected = t.score(X[900:])

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "m.ubj"
        kept = t.save(path)
        assert kept == t.best_iteration + 1

        booster = xgb.Booster()
        booster.load_model(str(path))
        assert booster.num_boosted_rounds() == kept
        got = booster.predict(xgb.DMatrix(X[900:]))

    assert np.allclose(expected, got, atol=1e-6)


def test_안_쓴_컬럼도_0으로_남는다():
    """빠뜨리면 컬럼 수가 달라져 나중에 모델끼리 비교할 때 어긋난다."""
    X, y = make_xy()
    X["안쓰임"] = 0.0
    t = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    imp = feature_importance(t, top=99)
    row = imp[imp["feature"] == "안쓰임"]
    assert len(row) == 1
    assert row.iloc[0]["splits"] == 0


# ── 나무 수를 고정해 전체로 다시 학습 ──────────────────────────────────────


def test_받은_나무_수를_그대로_쓴다():
    X, y = make_xy()
    t = refit_on_all(X, y, 7, CONFIG)
    assert t.best_iteration + 1 == 7
    assert t.model.get_booster().num_boosted_rounds() == 7


def test_다시_학습한_모델은_출처를_남긴다():
    """결과 파일만 보고 '나무 수를 어디서 정했나'를 알 수 있어야 한다."""
    X, y = make_xy()
    probe = train_xgb(X[:300], y[:300], X[300:], y[300:], CONFIG)
    assert probe.tree_source == "early_stop"
    assert probe.stop_rows == 100

    t = refit_on_all(X, y, probe.best_iteration + 1, CONFIG)
    assert t.tree_source == "fixed"
    assert t.stop_rows == 0
    assert t.fit_rows == len(X)


def test_다시_학습할_때는_조기_종료를_쓰지_않는다():
    """early_stopping_rounds가 남아 있으면 eval_set 없이 학습이 막힌다."""
    X, y = make_xy()
    t = refit_on_all(X, y, 5, CONFIG)
    assert "early_stopping_rounds" not in t.params


def test_나무_수가_0_이하면_거부한다():
    X, y = make_xy()
    for bad in (0, -3):
        with pytest.raises(ValueError, match="나무 수가 잘못됐습니다"):
            refit_on_all(X, y, bad, CONFIG)


def test_다시_학습할_때도_사기가_없으면_거부한다():
    X, y = make_xy()
    with pytest.raises(ValueError, match="사기 거래가 없"):
        refit_on_all(X, y * 0, 5, CONFIG)


def test_같은_시드면_다시_학습해도_같은_점수가_나온다():
    X, y = make_xy()
    a = refit_on_all(X, y, 9, CONFIG).score(X)
    b = refit_on_all(X, y, 9, CONFIG).score(X)
    assert np.array_equal(a, b)
