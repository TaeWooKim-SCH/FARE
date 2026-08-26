"""컬럼 프로파일 생성 진입점.

    python scripts/eda/run_profile.py            # 전체
    python scripts/eda/run_profile.py --nrows 50000   # 빠른 확인

결과는 `results/eda/`에 설정과 함께 저장한다. 설정 없는 결과는 남기지 않는다
(research-experiment 규약).

전체 기준과 학습셋 기준을 모두 낸다. 문서(부록 A·C)는 전체 기준이고, 무엇을 버릴지
같은 결정은 학습셋 기준이어야 한다. 둘이 갈리는 컬럼을 요약에 따로 뽑는 이유다.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# 윈도우 콘솔 기본 인코딩(cp949)은 한글 일부와 em-dash를 못 찍고 죽는다.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from src.data.loader import load_config, load_identity, load_transactions  # noqa: E402
from src.data.split import time_split  # noqa: E402
from src.eda.profile import (  # noqa: E402
    column_profile,
    constant_columns,
    missing_pattern_blocks,
    productcd_gating,
)


def build_summary(
    tx_full: pd.DataFrame,
    tx_train: pd.DataFrame,
    idf_full: pd.DataFrame,
    blocks: pd.DataFrame,
    gating: pd.DataFrame,
    n_rows: dict[str, int],
) -> str:
    const_full = set(constant_columns(tx_full))
    const_train = set(constant_columns(tx_train))
    only_train = sorted(const_train - const_full)

    lines = [
        "# EDA 프로파일 요약",
        "",
        f"거래 {n_rows['transaction']:,}행 · 학습셋 {n_rows['train']:,}행 · identity {n_rows['identity']:,}행",
        "",
        "## 컬럼 수",
        "",
        f"- 거래 feature: {len(tx_full)}개 (TransactionID·isFraud 제외)",
        f"- identity feature: {len(idf_full)}개",
        "",
        "## 상수 컬럼",
        "",
        f"- 전체 기준: {len(const_full)}개 {sorted(const_full) if const_full else ''}",
        f"- 학습셋 기준: {len(const_train)}개 {sorted(const_train) if const_train else ''}",
    ]
    if only_train:
        lines += [
            "",
            f"**학습셋에서만 상수인 컬럼: {only_train}**",
            "",
            "학습셋에서는 값이 하나뿐이라 모델이 아무것도 배울 수 없지만 이후 구간에서는 값이 갈린다.",
            "전체 기준으로 판정했다면 남겼을 컬럼이고, 그 판단 자체가 미래를 본 것이 된다.",
            "무엇을 버릴지는 반드시 학습셋 기준으로 정한다.",
        ]

    near = tx_full[tx_full["top_share"] >= 0.99]
    lines += [
        "",
        "## 거의 상수 (최빈값이 99% 이상)",
        "",
        f"- {len(near)}개",
        "",
        "## V 결측 패턴 블록",
        "",
        f"- 서로 다른 패턴: {len(blocks)}개",
        f"- 번호가 이어지지 않는 블록: {int((~blocks['contiguous']).sum())}개",
        "",
        "번호가 끊기는 블록이 있으면 `constraints.yaml`에 `V279-V321` 같은 구간으로 적을 수 없다.",
        "컬럼 목록으로 적어야 한다.",
        "",
        "| 컬럼 수 | 결측 | 연속 번호 | 시작 |",
        "|---|---|---|---|",
    ]
    for _, r in blocks.iterrows():
        head = r["columns"].split(",")[0]
        lines.append(
            f"| {r['n_columns']} | {r['missing_rate']:.1%} | {'예' if r['contiguous'] else '**아니오**'} | {head} |"
        )

    lines += [
        "",
        "## ProductCD가 가르는 컬럼",
        "",
        "| 상품 | 행 수 | 통째로 비어 있는 컬럼 수 |",
        "|---|---|---|",
    ]
    for _, r in gating.iterrows():
        lines.append(f"| {r['product']} | {r['rows']:,} | {r['n_all_missing']} |")
    lines += [
        "",
        "상품별로 존재하는 feature가 달라, 실험 결과를 상품별로 쪼개 보고해야 이 비대칭이 드러난다.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="IEEE-CIS 컬럼 프로파일 생성")
    parser.add_argument("--nrows", type=int, default=None, help="빠른 확인용으로 앞 N행만 읽는다")
    parser.add_argument("--out", default="results/eda", help="결과를 저장할 디렉터리")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    def say(message: str) -> None:
        print(message)
        log.append(message)

    cfg = load_config()
    say(f"거래 테이블 로딩 (nrows={args.nrows})")
    tx = load_transactions(cfg, nrows=args.nrows)
    say(f"identity 테이블 로딩")
    idf = load_identity(cfg)

    say("시간순 분할")
    split = time_split(tx, cfg)

    say("프로파일 계산 — 거래(전체)")
    tx_full = column_profile(tx)
    say("프로파일 계산 — 거래(학습셋)")
    tx_train = column_profile(split.train)
    say("프로파일 계산 — identity(전체)")
    idf_full = column_profile(idf)
    say("V 결측 패턴 블록")
    blocks = missing_pattern_blocks(tx)
    say("ProductCD 게이팅")
    gating = productcd_gating(tx)

    tx_full.to_csv(out_dir / "profile_transaction_full.csv", index=False, encoding="utf-8")
    tx_train.to_csv(out_dir / "profile_transaction_train.csv", index=False, encoding="utf-8")
    idf_full.to_csv(out_dir / "profile_identity_full.csv", index=False, encoding="utf-8")
    blocks.to_csv(out_dir / "v_blocks.csv", index=False, encoding="utf-8")
    gating.to_csv(out_dir / "productcd_gating.csv", index=False, encoding="utf-8")

    n_rows = {"transaction": len(tx), "train": len(split.train), "identity": len(idf)}
    summary = build_summary(tx_full, tx_train, idf_full, blocks, gating, n_rows)
    (out_dir / "summary.md").write_text(summary, encoding="utf-8")

    # 어떤 설정으로 뽑은 값인지 남긴다. 이게 없으면 수치를 다시 만들 수 없다.
    used = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
        "nrows": args.nrows,
        "rows": n_rows,
        "data_config": cfg,
    }
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(used, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    say(f"완료 — {out_dir}")
    (out_dir / "log.txt").write_text("\n".join(log), encoding="utf-8")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
