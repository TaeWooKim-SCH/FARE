"""지표와 임계값 결정 검증.

여기서 틀리면 "공격 전후로 얼마나 나빠졌나"라는 이 연구의 주 결과가 통째로 틀린다.
특히 τ를 어디서 정하느냐가 중요하다 — 평가셋을 보고 정하면 평가셋이 학습에 쓰인 것이 된다.
"""

import numpy as np
import pytest

from src.models.metrics import choose_threshold, evaluate_at, f_beta, rank_metrics


# ── F-베타 ─────────────────────────────────────────────────────────────────


def test_precision과_recall이_같으면_베타와_무관하게_같은_값이_나온다():
    for beta in (0.5, 1.0, 2.0):
        assert f_beta(np.array([0.6]), np.array([0.6]), beta)[0] == pytest.approx(0.6)


def test_베타가_2면_recall이_높은_쪽을_더_쳐준다():
    recall_높음 = f_beta(np.array([0.2]), np.array([0.8]), 2.0)[0]
    precision_높음 = f_beta(np.array([0.8]), np.array([0.2]), 2.0)[0]
    assert recall_높음 > precision_높음


def test_베타가_1이면_둘을_똑같이_친다():
    a = f_beta(np.array([0.2]), np.array([0.8]), 1.0)[0]
    b = f_beta(np.array([0.8]), np.array([0.2]), 1.0)[0]
    assert a == pytest.approx(b)


def test_둘_다_0이면_0을_준다():
    """0으로 나누면 nan이 나오고, nan은 argmax에서 조용히 밀려나 엉뚱한 τ가 잡힌다."""
    assert f_beta(np.array([0.0]), np.array([0.0]), 2.0)[0] == 0.0


# ── 임계값 결정 ────────────────────────────────────────────────────────────


def test_완전히_갈리면_사기만_잡는_임계값을_고른다():
    y = np.array([0, 0, 0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.2, 0.3, 0.9, 0.95])
    t = choose_threshold(y, score)
    assert t.recall == 1.0
    assert t.precision == 1.0
    assert t.flagged == 2


def test_고른_임계값이_실제로_F베타를_최대로_만든다():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.05, 3000)
    score = np.clip(y * 0.4 + rng.normal(0.3, 0.2, 3000), 0, 1)
    t = choose_threshold(y, score, beta=2.0)

    best = evaluate_at(y, score, t.tau)["f2"]
    for other in np.linspace(score.min(), score.max(), 200):
        assert evaluate_at(y, score, other)["f2"] <= best + 1e-9


def test_베타를_올리면_임계값이_낮아진다():
    """Recall을 더 치면 더 많이 표시하게 된다. 방향이 반대면 어딘가 틀린 것이다."""
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.05, 3000)
    score = np.clip(y * 0.35 + rng.normal(0.3, 0.2, 3000), 0, 1)
    assert choose_threshold(y, score, beta=4.0).tau <= choose_threshold(y, score, beta=0.5).tau


def test_사기가_하나도_없으면_거부한다():
    with pytest.raises(ValueError, match="사기 거래가 하나도 없"):
        choose_threshold(np.zeros(10), np.linspace(0, 1, 10))


def test_표시_비율을_함께_돌려준다():
    """심사팀이 감당할 양인지 봐야 해서, Recall만으로는 부족하다."""
    y = np.array([0, 0, 0, 1])
    t = choose_threshold(y, np.array([0.1, 0.2, 0.3, 0.9]))
    assert t.flag_rate == t.flagged / 4


# ── 순위 지표 ──────────────────────────────────────────────────────────────


def test_완벽히_갈리면_둘_다_1이_된다():
    y = np.array([0, 0, 1, 1])
    r = rank_metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))
    assert r["roc_auc"] == 1.0
    assert r["pr_auc"] == 1.0


def test_순위만_보므로_점수를_늘려도_안_바뀐다():
    """PR-AUC와 ROC-AUC는 임계값과 무관하다. 스케일을 바꿔도 같아야 한다."""
    y = np.array([0, 1, 0, 1, 1, 0])
    score = np.array([0.1, 0.7, 0.3, 0.6, 0.9, 0.2])
    a = rank_metrics(y, score)
    b = rank_metrics(y, score * 100 + 5)
    assert a["roc_auc"] == pytest.approx(b["roc_auc"])
    assert a["pr_auc"] == pytest.approx(b["pr_auc"])


def test_사기율을_함께_남긴다():
    """PR-AUC는 사기율이 다르면 그대로 비교하면 안 된다. 분모를 같이 적어둔다."""
    assert rank_metrics(np.array([0, 0, 0, 1]), np.array([0.1, 0.2, 0.3, 0.9]))["positive_rate"] == 0.25


# ── 정해진 임계값으로 재기 ─────────────────────────────────────────────────


def test_넘겨준_임계값을_그대로_쓴다():
    """여기서 τ를 다시 고르면 평가셋을 보고 정하는 셈이 된다."""
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.6, 0.9])
    느슨 = evaluate_at(y, score, 0.5)
    빡빡 = evaluate_at(y, score, 0.8)
    assert 느슨["recall"] == 1.0
    assert 빡빡["recall"] == 0.5


def test_네_칸을_정확히_센다():
    y = np.array([1, 1, 0, 0, 0])
    score = np.array([0.9, 0.2, 0.8, 0.1, 0.1])
    m = evaluate_at(y, score, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["recall"] == 0.5
    assert m["precision"] == 0.5


def test_임계값이_너무_높으면_0을_준다():
    """아무것도 표시 안 하면 precision이 0/0이 된다. nan 대신 0으로 둔다."""
    m = evaluate_at(np.array([0, 1]), np.array([0.1, 0.2]), 0.99)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0


def test_경계값은_표시하는_쪽에_넣는다():
    """점수가 τ와 정확히 같을 때 어느 쪽인지 정해두지 않으면 결과가 흔들린다."""
    assert evaluate_at(np.array([1]), np.array([0.5]), 0.5)["tp"] == 1


def test_베타를_바꾸면_결과_키_이름도_바뀐다():
    assert "f2" in evaluate_at(np.array([0, 1]), np.array([0.1, 0.9]), 0.5, beta=2.0)
    assert "f1" in evaluate_at(np.array([0, 1]), np.array([0.1, 0.9]), 0.5, beta=1.0)
