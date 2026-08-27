"""C·D에 미래가 섞였는지 확인하는 진입점.

    python scripts/eda/check_leakage.py

결과는 `results/eda/`에 저장한다. feature-taxonomy.md 5절에 적어둔 수치가 여기서 나온다.
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

# 윈도우 콘솔 기본 인코딩(cp949)은 한글 일부를 못 찍고 죽는다.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

from src.data.loader import load_config, load_transactions  # noqa: E402
from src.eda.leakage import direction_share, first_row_reveals_total, tail_median_ratio  # noqa: E402

COUNT_COLUMNS = [f"C{i}" for i in range(1, 15)]
WATCH = ["C1", "C13", "C14"]
TIME_COLUMNS = ["D1", "D3", "D15"]


def main() -> int:
    parser = argparse.ArgumentParser(description="C·D에 미래가 섞였는지 확인")
    parser.add_argument("--out", default="results/eda", help="결과를 저장할 디렉터리")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    needed = ["TransactionDT", "isFraud", "card1", "addr1"] + COUNT_COLUMNS + TIME_COLUMNS
    tx = load_transactions(cfg, columns=sorted(set(needed)))
    print(f"거래 {len(tx):,}건을 읽었다")

    reveals = first_row_reveals_total(tx, COUNT_COLUMNS)
    tail = tail_median_ratio(tx, WATCH + TIME_COLUMNS)
    direction = direction_share(tx, WATCH)

    reveals.to_csv(out_dir / "leakage_first_row.csv", index=False, encoding="utf-8")
    tail.to_csv(out_dir / "leakage_tail.csv", index=False, encoding="utf-8")
    direction.to_csv(out_dir / "leakage_direction.csv", index=False, encoding="utf-8")

    lines = [
        "# C·D에 미래가 섞였는지",
        "",
        "계산 방법이 공개되지 않아 직접 확인은 못 한다. 간접 신호 셋을 본다.",
        "",
        "## 1. 카드의 첫 거래에서 이미 총 거래 수가 보이나",
        "",
        "미래까지 셌다면 첫 거래에서도 그 카드의 총 거래 수가 보여야 한다.",
        f"거래 3건 이상인 카드 {int(reveals['cards'].iloc[0]):,}장을 봤다.",
        "",
        "| 컬럼 | 총 거래 수와 같은 비율 |",
        "|---|---|",
    ]
    for _, r in reveals.iterrows():
        lines.append(f"| {r['column']} | {r['match_rate']:.3%} |")
    lines += [
        "",
        "비율이 1에 가까우면 미래를 본 것이고, 0에 가까우면 과거만 센 것이다.",
        "",
        "## 2. 마지막 날에 값이 꺾이나",
        "",
        "| 컬럼 | 전체 중앙값 | 마지막 날 중앙값 | 비율 |",
        "|---|---|---|---|",
    ]
    for _, r in tail.iterrows():
        lines.append(
            f"| {r['column']} | {r['median_all']:.1f} | {r['median_tail']:.1f} | {r['ratio']:.2f} |"
        )
    lines += [
        "",
        "D1이 뒤로 갈수록 커지는 것은 시간이 흐르며 카드가 오래되는 자연스러운 결과다.",
        "",
        "## 3. 같은 카드 안에서 값이 늘기만 하나",
        "",
        "| 컬럼 | 늘어남 | 그대로 | 줄어듦 |",
        "|---|---|---|---|",
    ]
    for _, r in direction.iterrows():
        lines.append(f"| {r['column']} | {r['up']:.1%} | {r['same']:.1%} | {r['down']:.1%} |")
    lines += [
        "",
        "줄어드는 것은 우리가 카드를 정확히 못 짚어 서로 다른 카드가 섞인 탓으로 보인다.",
        "",
    ]
    summary = "\n".join(lines)
    (out_dir / "leakage_summary.md").write_text(summary, encoding="utf-8")

    used = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pandas": pd.__version__,
        "python": sys.version.split()[0],
        "rows": len(tx),
        "data_config": cfg,
    }
    (out_dir / "leakage_config.yaml").write_text(
        yaml.safe_dump(used, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    print(f"완료 — {out_dir}")
    print()
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
