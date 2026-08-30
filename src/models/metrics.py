"""탐지 성능 지표와 운영 임계값 결정.

정확도는 쓰지 않는다. 사기가 3.5%라서 전부 정상이라고 답해도 96.5%가 나온다.

임계값을 쓰는 지표(Recall, Precision)와 안 쓰는 지표(PR-AUC, ROC-AUC)를 나눠 둔다.
앞의 것은 τ를 어디에 두느냐에 따라 통째로 달라지므로 τ 없이는 의미가 없고, 뒤의 것은
순위만 보므로 τ와 무관하게 모델끼리 비교할 수 있다.

τ는 **검증셋에서만** 정한다. 평가셋을 보고 정하면 그 순간 평가셋이 학습에 쓰인 것이 된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

# 사기를 놓치는 쪽이 헛짚는 쪽보다 비싸다는 판단. research-plan.md 4.3의 결정을 따른다.
DEFAULT_BETA = 2.0


def f_beta(precision: np.ndarray, recall: np.ndarray, beta: float = DEFAULT_BETA) -> np.ndarray:
    """F-베타. beta가 크면 Recall을 더 쳐준다.

    분모가 0이 되는 지점(둘 다 0)은 0으로 둔다. 그런 임계값은 어차피 고를 일이 없다.
    """
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.divide((1 + b2) * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)


@dataclass(frozen=True)
class Threshold:
    """검증셋에서 고른 운영 임계값과, 그 자리에서의 성능."""

    tau: float
    beta: float
    precision: float
    recall: float
    f_beta: float
    flagged: int
    n: int

    @property
    def flag_rate(self) -> float:
        """전체 거래 중 사기로 표시하는 비율. 심사팀이 감당할 양인지 보는 값."""
        return self.flagged / self.n


def choose_threshold(y_true, score, beta: float = DEFAULT_BETA) -> Threshold:
    """F-베타를 가장 크게 만드는 임계값을 찾는다. **검증셋에만 쓴다.**

    후보를 임의로 나누지 않고 `precision_recall_curve`가 주는 지점을 전부 본다.
    점수가 바뀌는 자리마다 하나씩 나오므로 이보다 촘촘하게 볼 필요가 없다.
    """
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    if y_true.sum() == 0:
        raise ValueError("사기 거래가 하나도 없습니다. 임계값을 정할 수 없습니다.")

    precision, recall, thresholds = precision_recall_curve(y_true, score)
    # precision_recall_curve는 마지막에 (precision=1, recall=0)을 덧붙이는데
    # 여기에 대응하는 임계값이 없다. 길이를 맞추려면 잘라내야 한다.
    precision, recall = precision[:-1], recall[:-1]

    scores = f_beta(precision, recall, beta)
    best = int(np.argmax(scores))
    tau = float(thresholds[best])

    return Threshold(
        tau=tau,
        beta=beta,
        precision=float(precision[best]),
        recall=float(recall[best]),
        f_beta=float(scores[best]),
        flagged=int((score >= tau).sum()),
        n=len(score),
    )


def rank_metrics(y_true, score) -> dict[str, float]:
    """임계값과 무관한 지표. 모델끼리 비교하거나 캐글 점수와 대볼 때 쓴다."""
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)
    return {
        # 캐글 IEEE-CIS 대회가 ROC-AUC로 순위를 매겼다. 대조하려면 이게 필요하다.
        "roc_auc": float(roc_auc_score(y_true, score)),
        # 불균형에서는 PR-AUC가 실제 쓸모를 더 잘 보여준다. 우리 주 비교 지표다.
        "pr_auc": float(average_precision_score(y_true, score)),
        "positive_rate": float(y_true.mean()),
        "n": int(len(y_true)),
    }


def evaluate_at(y_true, score, tau: float, beta: float = DEFAULT_BETA) -> dict[str, float]:
    """검증셋에서 정한 τ를 그대로 써서 잰다. τ를 여기서 다시 고르지 않는다."""
    y_true = np.asarray(y_true)
    predicted = np.asarray(score, dtype=float) >= tau

    tp = int((predicted & (y_true == 1)).sum())
    fp = int((predicted & (y_true == 0)).sum())
    fn = int((~predicted & (y_true == 1)).sum())
    tn = int((~predicted & (y_true == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "tau": float(tau),
        "recall": recall,
        "precision": precision,
        f"f{beta:g}": float(f_beta(np.array([precision]), np.array([recall]), beta)[0]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        # 심사팀이 하루에 볼 수 있는 양을 넘지 않는지 보는 값
        "flag_rate": (tp + fp) / len(y_true),
    }
