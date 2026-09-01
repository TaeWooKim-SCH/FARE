"""하이퍼파라미터 탐색 검증.

성능이 아니라 **절차**를 본다. 같은 시드면 같은 설정이 뽑히는지, 중간에 끊겨도 이어지는지,
검증셋에 안 닿는지.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.models.search import load_done, put, sample_params, search

SPACE = {
    "max_depth": [4, 6, 8],
    "hidden_layer_sizes": [[64], [128, 64]],
    "learning_rate": {"log": [0.001, 0.1]},
    "subsample": {"uniform": [0.6, 1.0]},
}
SEARCH_CFG = {"seed": 42, "trials": 4, "metric": "pr_auc"}
MODEL_CFG = {"seed": 42, "mlp": {"network": {"max_iter": 60, "alpha": 0.1}}, "xgboost": {"max_depth": 8}}


# ── 설정 뽑기 ────────────────────────────────────────────────────────────────


def test_같은_시드면_같은_설정이_뽑힌다():
    """다시 돌렸을 때 n번째 시행이 같은 설정이어야 이어서 돌 수 있다."""
    a = [sample_params(SPACE, np.random.default_rng(42)) for _ in range(3)]
    b = [sample_params(SPACE, np.random.default_rng(42)) for _ in range(3)]
    assert a == b


def test_은닉층_같은_목록은_망가지지_않고_뽑힌다():
    """rng.choice는 리스트 안의 리스트를 배열로 펴버려서 [256, 64]가 깨진다."""
    뽑힌것 = [sample_params(SPACE, np.random.default_rng(s))["hidden_layer_sizes"] for s in range(20)]
    assert all(v in ([64], [128, 64]) for v in 뽑힌것)


def test_로그_눈금은_자릿수마다_고르게_뽑는다():
    """학습률을 고르게 뽑으면 큰 값 쪽만 잔뜩 보게 된다. 0.001~0.01 구간과
    0.01~0.1 구간이 비슷하게 나와야 한다.
    """
    rng = np.random.default_rng(0)
    값 = [sample_params(SPACE, rng)["learning_rate"] for _ in range(400)]
    아래 = sum(1 for v in 값 if v < 0.01)
    assert 0.35 < 아래 / len(값) < 0.65
    assert all(0.001 <= v <= 0.1 for v in 값)


def test_균등은_범위_안에서만_뽑는다():
    rng = np.random.default_rng(0)
    값 = [sample_params(SPACE, rng)["subsample"] for _ in range(100)]
    assert all(0.6 <= v <= 1.0 for v in 값)


# ── config에 얹기 ────────────────────────────────────────────────────────────


def test_설정을_얹어도_원본이_안_바뀐다():
    """원본이 바뀌면 다음 시행이 앞 시행의 설정을 물려받는다."""
    얹은것 = put(MODEL_CFG, ("mlp", "network"), {"alpha": 0.5})
    assert 얹은것["mlp"]["network"]["alpha"] == 0.5
    assert MODEL_CFG["mlp"]["network"]["alpha"] == 0.1


def test_얹지_않은_값은_그대로_남는다():
    얹은것 = put(MODEL_CFG, ("mlp", "network"), {"alpha": 0.5})
    assert 얹은것["mlp"]["network"]["max_iter"] == 60
    assert 얹은것["seed"] == 42


def test_한_겹짜리_자리에도_얹힌다():
    얹은것 = put(MODEL_CFG, ("xgboost",), {"max_depth": 3})
    assert 얹은것["xgboost"]["max_depth"] == 3


# ── 탐색 루프 ────────────────────────────────────────────────────────────────


class 점수내는_스텁:
    """`max_depth`가 클수록 좋은 점수가 나오는 가짜 모델. 최선을 미리 알 수 있다."""

    def __init__(self, depth):
        self.depth = depth
        self.rounds = depth * 10

    def score(self, X):
        # depth가 클수록 정답과 잘 맞는 점수를 낸다
        return np.where(X["y"].to_numpy() == 1, self.depth / 10, 0.0)


def 가짜학습(X_fit, y_fit, X_stop, y_stop, cfg):
    return 점수내는_스텁(cfg["xgboost"]["max_depth"])


@pytest.fixture
def 조각():
    y = pd.Series([0, 1] * 25)
    X = pd.DataFrame({"y": y})
    return {"fit": X, "stop": X}, {"fit": y, "stop": y}


def test_시행마다_결과를_파일에_이어_쓴다(조각, tmp_path):
    X, y = 조각
    search("가짜", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    줄 = load_done(tmp_path / "가짜.jsonl")
    assert len(줄) == SEARCH_CFG["trials"]
    assert [r["trial"] for r in 줄] == [0, 1, 2, 3]
    assert all("pr_auc" in r and "seconds" in r and "params" in r for r in 줄)


def test_끊겼다_다시_돌리면_이어서_한다(조각, tmp_path):
    """열몇 시간짜리 실행이 죽어서 처음부터 다시 하는 일이 없어야 한다."""
    X, y = 조각
    앞부분 = {**SEARCH_CFG, "trials": 2}
    search("가짜", SPACE, 앞부분, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    assert len(load_done(tmp_path / "가짜.jsonl")) == 2

    search("가짜", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    줄 = load_done(tmp_path / "가짜.jsonl")
    assert [r["trial"] for r in 줄] == [0, 1, 2, 3]


def test_이어서_돈_설정이_처음부터_돈_것과_같다(조각, tmp_path):
    """건너뛴 시행도 난수를 소비해야 n번째가 언제나 같은 설정이 된다."""
    X, y = 조각
    search("한번에", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    search("나눠서", SPACE, {**SEARCH_CFG, "trials": 2}, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    search("나눠서", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)

    한번에 = [r["params"] for r in load_done(tmp_path / "한번에.jsonl")]
    나눠서 = [r["params"] for r in load_done(tmp_path / "나눠서.jsonl")]
    assert 한번에 == 나눠서


def test_제일_좋은_설정을_고른다(조각, tmp_path):
    X, y = 조각
    결과 = search("가짜", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    assert 결과["best"]["pr_auc"] == max(r["pr_auc"] for r in 결과["all"])
    assert 결과["trials_done"] == 4


def test_검증셋에_안_닿는다(조각, tmp_path):
    """설정을 검증셋에서 고르면 τ를 정할 데이터가 이미 모델을 고르는 데 쓰인 셈이 된다."""
    X, y = 조각
    X = {**X, "val": pd.DataFrame({"y": [999] * 10}), "train": X["fit"]}
    y = {**y, "val": pd.Series([1] * 10), "train": y["fit"]}
    결과 = search("가짜", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    assert 결과["chosen_on"] == "stop"


def test_탐색_범위를_결과에_남긴다(조각, tmp_path):
    """이 기록이 그대로 논문 표가 된다 (research-plan.md 4.2 ④)."""
    X, y = 조각
    결과 = search("가짜", SPACE, SEARCH_CFG, MODEL_CFG, ("xgboost",), 가짜학습, X, y, tmp_path)
    assert 결과["space"] == SPACE
    assert 결과["trials_planned"] == 4
    assert json.dumps(결과, ensure_ascii=False)  # 결과 파일로 나가므로 직렬화가 돼야 한다
