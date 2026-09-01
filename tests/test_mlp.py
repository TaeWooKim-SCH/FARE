"""MLP 학습 코드 검증.

성능이 잘 나오는지가 아니라, **평가셋이 학습에 닿지 않는지**와 **설정이 기록되는지**를 본다.
성능은 실제 데이터로만 의미가 있어서 여기서 재지 않는다.

가장 중요한 검사는 `test_조기_종료를_config에서_켜도_꺼진다`이다. sklearn의
`early_stopping=True`는 내부에서 무작위 분할을 해서 이 연구의 절대 규칙을 조용히 어긴다.
"""

import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from src.models.mlp import loss_curve, train_mlp


def config(**network):
    """작은 신경망 설정. 테스트마다 필요한 값만 바꿔 쓴다."""
    base = {
        "hidden_layer_sizes": [8],
        "alpha": 0.0001,
        "learning_rate_init": 0.01,
        "batch_size": 64,
        "max_iter": 30,
        "n_iter_no_change": 5,
        "tol": 1e-4,
    }
    return {"seed": 42, "mlp": {"network": {**base, **network}}}


CONFIG = config()


def make_xy(n=400, seed=0):
    """MlpAdapter를 거친 뒤의 모양을 흉내 낸다 — float32에 결측이 없다."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)}
    ).astype("float32")
    y = pd.Series((X["a"] + rng.normal(0, 0.3, n) > 0.8).astype(int))
    return X, y


def test_학습이_되고_점수가_확률로_나온다():
    X, y = make_xy()
    s = train_mlp(X, y, CONFIG).score(X)
    assert len(s) == len(X)
    assert ((s >= 0) & (s <= 1)).all()


def test_조기_종료를_config에서_켜도_꺼진다():
    """sklearn 조기 종료는 train_test_split으로 검증 조각을 뗀다. 무작위 분할이라
    시간이 섞이고, 코드에 아무 표시도 안 난 채 미래가 과거로 샌다.
    """
    X, y = make_xy()
    t = train_mlp(X, y, config(early_stopping=True))
    assert t.params["early_stopping"] is False
    assert t.model.early_stopping is False


def test_검증셋과_평가셋을_넘길_자리가_없다():
    """인자가 학습셋 하나뿐이라, 검증셋을 넣으려면 함수를 고쳐야 한다."""
    import inspect

    assert list(inspect.signature(train_mlp).parameters) == ["X_train", "y_train", "config"]


def test_설정과_행_수를_기록한다():
    """결과 파일에 같이 적어야 나중에 같은 숫자를 다시 만들 수 있다."""
    X, y = make_xy()
    t = train_mlp(X, y, CONFIG)
    assert t.params["random_state"] == 42
    assert t.params["hidden_layer_sizes"] == (8,)  # yaml 리스트를 튜플로 바꿔 넘긴다
    assert t.fit_rows == len(X)
    assert t.feature_names == ("a", "b", "c")


def test_float32_입력이_float32로_남는다():
    """41만 행 543컬럼이면 float64로 올라갈 때 1.8GB인데 float32면 898MB다."""
    X, y = make_xy()
    assert (X.dtypes == "float32").all()
    t = train_mlp(X, y, CONFIG)
    assert t.model.coefs_[0].dtype == np.float32


def test_같은_시드면_같은_점수가_나온다():
    X, y = make_xy()
    a = train_mlp(X, y, CONFIG).score(X)
    b = train_mlp(X, y, CONFIG).score(X)
    assert np.array_equal(a, b)


def test_몇_바퀴_돌았고_왜_멈췄는지_남긴다():
    """max_iter에 닿아 멈춘 것과 손실이 평평해져 멈춘 것은 뜻이 다르다."""
    X, y = make_xy()
    짧게 = train_mlp(X, y, config(max_iter=3, n_iter_no_change=99))
    assert 짧게.rounds == 3
    assert 짧게.stopped_early is False

    길게 = train_mlp(X, y, config(max_iter=500, n_iter_no_change=3, tol=0.01))
    assert 길게.rounds < 500
    assert 길게.stopped_early is True


def test_학습셋에_사기가_없으면_거부한다():
    """전부 정상이면 학습이 되긴 하는데 아무 뜻이 없는 모델이 나온다."""
    X, y = make_xy()
    with pytest.raises(ValueError, match="학습셋에 사기 거래가 없"):
        train_mlp(X, y * 0, CONFIG)


def test_점수를_낼_때_컬럼_순서가_다르면_거부한다():
    """순서만 바뀌어도 모델은 조용히 엉뚱한 값을 읽는다."""
    X, y = make_xy()
    t = train_mlp(X, y, CONFIG)
    with pytest.raises(ValueError, match="컬럼 구성이 학습 때와 다릅니다"):
        t.score(X[["b", "a", "c"]])


def test_저장한_파일을_불러도_같은_점수가_나온다():
    """공격 단계에서 이 파일을 불러다 쓰므로, 어긋나면 회피율이 미묘하게 안 맞는다."""
    X, y = make_xy()
    t = train_mlp(X, y, CONFIG)
    expected = t.score(X)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "net.joblib"
        size = t.save(path)
        assert size == path.stat().st_size > 0
        got = joblib.load(path).predict_proba(X)[:, 1]

    assert np.array_equal(expected, got)


def test_손실_곡선을_바퀴마다_남긴다():
    """성능 곡선이 아니라 학습 손실이다. 덜 돌았는지 보는 용도다."""
    X, y = make_xy()
    t = train_mlp(X, y, CONFIG)
    curve = loss_curve(t)
    assert list(curve.columns) == ["round", "loss"]
    assert len(curve) == t.rounds
    assert curve["loss"].iloc[-1] == pytest.approx(t.final_loss)
