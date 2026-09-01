"""XGBoost 기준선의 PR 곡선과 ROC 곡선을 그린다.

실행:
    python scripts/models/plot_curves.py

**모델을 다시 학습하지 않는다.** `results/models/xgboost_baseline.ubj`를 불러 점수만 낸다.
τ도 같은 실행에서 나온 `xgboost_baseline.json`에서 읽는다. 그래서 그림의 숫자가 기준선
파일의 숫자와 반드시 같다.

평가셋을 읽지만 여기서 고르는 것이 없다. 기준선을 낼 때 이미 연 데이터를 그림으로 옮기는
것뿐이라, 사람 손을 거쳐 평가셋이 선택셋이 되는 경로가 아니다.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
from matplotlib import font_manager
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.data.loader import REPO_ROOT
from src.models.metrics import evaluate_at
from src.models.runner import OUT_DIR, prepare, short_path

FIG_DIR = REPO_ROOT / "results" / "figures"

# dataviz 참조 팔레트. 1번 슬롯(파랑)과 2번 슬롯(주황)은 색각 이상에서도 갈리는 조합으로
# 이미 검증된 짝이다. 두 계열뿐이라 여기서 색을 새로 만들 이유가 없다.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SUB = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# 제일 중요한 평가셋에 1번 슬롯을 준다. 학습셋은 성능이 아니라 "얼마나 외웠나"를 보는
# 참고선이라 3번 슬롯에 선도 얇게 둔다. 3번(청록)은 밝은 바탕에서 대비가 낮아 곡선 옆에
# 이름을 직접 붙여야 한다.
SERIES = {"train": "#1baf7a", "val": "#eb6834", "test": "#2a78d6"}
LABEL = {"train": "학습셋", "val": "검증셋", "test": "평가셋"}
PARTS = ("train", "val", "test")


def use_korean_font() -> None:
    """한글이 깨지면 그림을 발표에 못 쓴다. 있는 글꼴 중 첫 번째를 잡는다."""
    있는것 = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "NanumGothic", "AppleGothic", "Gulim"):
        if name in 있는것:
            plt.rcParams["font.family"] = name
            break
    else:
        print("  경고: 한글 글꼴을 못 찾았다. 축 이름이 깨질 수 있다")
    # 한글 글꼴에는 유니코드 음수 기호가 없어서 마이너스가 네모로 나온다.
    plt.rcParams["axes.unicode_minus"] = False


def dress(ax) -> None:
    """축을 뒤로 물린다. 눈이 먼저 가야 할 곳은 곡선이다."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # 두 축이 같은 0~1 비율이라 정사각형으로 둔다. ROC의 대각선이 정확히 45도가 되어
    # 그 위에 붙이는 글자 각도를 맞출 수 있다.
    ax.set_aspect("equal")


def mark(ax, x, y, color, text=None, up=True):
    """운영점을 찍는다. 곡선과 겹치므로 흰 테두리를 둘러 떼어 놓는다."""
    ax.plot(x, y, "o", markersize=9, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    if text:
        # 두 점이 가까워서 한쪽은 위로, 한쪽은 아래로 뗀다. 글자가 곡선 위에 얹히므로
        # 바탕색 상자를 깔아 읽히게 한다.
        ax.annotate(
            text, (x, y), textcoords="offset points",
            xytext=(13, 8 if up else -30), fontsize=9, color=INK_SUB, zorder=6,
            # 상자를 불투명하게 둔다. 곡선 2cm를 가리는 것보다 숫자가 읽히는 쪽이 낫다.
            bbox=dict(boxstyle="round,pad=0.3", facecolor=SURFACE, edgecolor="none"),
        )


def main() -> None:
    use_korean_font()
    baseline = json.loads((OUT_DIR / "xgboost_baseline.json").read_text(encoding="utf-8"))
    tau, beta = baseline["threshold"]["tau"], baseline["threshold"]["beta"]

    p = prepare(final=True, stop_split=False)
    booster = xgb.Booster()
    booster.load_model(str(OUT_DIR / "xgboost_baseline.ubj"))
    print(f"불러온 모델 {booster.num_boosted_rounds():,}그루 / τ = {tau:.6f}")

    score = {n: booster.predict(xgb.DMatrix(p.X[n])) for n in PARTS}
    y = {n: p.y[n].to_numpy() for n in PARTS}

    # 두 패널이 정사각형이라 가로가 세로의 두 배쯤이어야 여백이 안 남는다.
    fig, (pr_ax, roc_ax) = plt.subplots(1, 2, figsize=(10, 5.4), facecolor=SURFACE)

    for name in PARTS:
        color = SERIES[name]
        # 학습셋은 이미 본 데이터라 성능이 아니다. 선을 얇게 해서 뒤로 물린다.
        얇게 = name == "train"
        꼬리 = " (이미 본 데이터)" if 얇게 else ""

        precision, recall, _ = precision_recall_curve(y[name], score[name])
        pr_auc = average_precision_score(y[name], score[name])
        pr_ax.plot(recall, precision, linewidth=1.4 if 얇게 else 2, color=color,
                   zorder=2 if 얇게 else 3,
                   label=f"{LABEL[name]} · PR-AUC {pr_auc:.3f}{꼬리}")

        fpr, tpr, _ = roc_curve(y[name], score[name])
        roc_ax.plot(fpr, tpr, linewidth=1.4 if 얇게 else 2, color=color,
                    zorder=2 if 얇게 else 3,
                    label=f"{LABEL[name]} · ROC-AUC {roc_auc_score(y[name], score[name]):.3f}{꼬리}")

        if 얇게:
            # 밝은 바탕에서 대비가 낮은 색이라 곡선 옆에 이름을 직접 붙인다.
            자리 = len(recall) // 2
            pr_ax.annotate("학습셋", (recall[자리], precision[자리]), fontsize=9, color=color,
                           textcoords="offset points", xytext=(6, 8), zorder=4)
            continue

        # 운영점은 검증셋과 평가셋에만 찍는다. 학습셋에서는 τ가 아무 뜻이 없다.
        at_tau = evaluate_at(y[name], score[name], tau, beta)
        mark(pr_ax, at_tau["recall"], at_tau["precision"], color,
             f"Recall {at_tau['recall']:.3f}\nPrecision {at_tau['precision']:.3f}",
             up=(name == "val"))
        mark(roc_ax, at_tau["fp"] / (at_tau["fp"] + at_tau["tn"]), at_tau["recall"], color)

    # 아무 능력 없이 찍었을 때의 선. 이게 있어야 0.55가 좋은 값인지 읽을 수 있다.
    사기율 = float(y["test"].mean())
    pr_ax.axhline(사기율, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    pr_ax.annotate(f"무작위로 찍으면 {사기율:.3f}", (0.02, 사기율), textcoords="offset points",
                   xytext=(0, 6), fontsize=9, color=MUTED)
    roc_ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    # 축이 정사각형이라 대각선이 정확히 45도다. 글자를 그 각도에 맞춰 얹는다.
    roc_ax.annotate("무작위로 찍으면 이 선", (0.62, 0.62), textcoords="offset points",
                    xytext=(0, -18), fontsize=9, color=MUTED, rotation=45,
                    rotation_mode="anchor", ha="center")

    pr_ax.set_title("PR 곡선 — 불균형에서 보는 지표", fontsize=12, color=INK, pad=10, loc="left")
    pr_ax.set_xlabel("Recall — 실제 사기 중 잡은 비율", fontsize=10, color=INK_SUB)
    pr_ax.set_ylabel("Precision — 찍은 것 중 진짜", fontsize=10, color=INK_SUB)
    roc_ax.set_title("ROC 곡선 — 캐글 점수와 견주는 값", fontsize=12, color=INK, pad=10, loc="left")
    roc_ax.set_xlabel("거짓 양성 비율 — 정상을 사기라 찍은 비율", fontsize=10, color=INK_SUB)
    roc_ax.set_ylabel("Recall", fontsize=10, color=INK_SUB)

    for ax, where in ((pr_ax, "upper right"), (roc_ax, "lower right")):
        dress(ax)
        legend = ax.legend(loc=where, fontsize=9, frameon=False)
        for text in legend.get_texts():
            text.set_color(INK_SUB)

    fig.suptitle(f"XGBoost 공격 전 기준선 (운영 임계값 τ = {tau:.3f})",
                 fontsize=13, color=INK, x=0.01, ha="left", y=0.975)
    # 점이 무엇인지 한 번만 적는다. 두 패널에 같은 뜻이라 각각 적으면 군더더기다.
    fig.text(0.01, 0.918, "● 표시는 τ에서 실제로 운영하는 자리다",
             fontsize=9.5, color=MUTED, ha="left")
    # tight_layout을 안 쓴다. 축을 정사각형으로 묶어두면 tight_layout이 크기를 줄이기
    # 전의 자리로 계산해서 x축 이름이 그림 밖으로 밀려난다. 여백을 직접 잡는다.
    # 위아래를 0.85와 0.10으로 두면 축 높이가 4.05인치가 되고, 정사각형 두 개에 사이
    # 간격까지 9.1인치라 가로도 딱 맞는다.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.85, bottom=0.10, wspace=0.22)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        out = FIG_DIR / f"xgboost_curves.{suffix}"
        fig.savefig(out, dpi=200, facecolor=SURFACE)
        print(f"저장: {short_path(out)}")


if __name__ == "__main__":
    main()
