"""XGBoost 탐지 모델을 학습하고 공격 전 기준선을 잰다.

실행:
    python scripts/models/train_xgb.py            # 검증셋까지만 본다
    python scripts/models/train_xgb.py --final    # 평가셋 수치까지 낸다

데이터 준비와 결과 보고는 `src/models/runner.py`가 세 모델에 똑같이 해준다. 이 파일에
남은 것은 XGBoost에만 있는 일, 즉 나무 수를 정하고 전체로 다시 학습하는 두 단계뿐이다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.runner import OUT_DIR, prepare, report, short_path
from src.models.xgb import feature_importance, refit_on_all, train_xgb


def main(final: bool) -> None:
    p = prepare(final=final, stop_split=True)
    X, y, model_cfg = p.X, p.y, p.model_cfg

    # 1단계: 나무를 몇 그루 쌓을지만 정한다. 검증셋은 안 쓴다.
    print("나무 수를 정하는 중...")
    probe = train_xgb(X["fit"], y["fit"], X["stop"], y["stop"], model_cfg)
    trees = probe.best_iteration + 1
    print(f"  {trees:,}그루 (최대 {model_cfg['xgboost']['n_estimators']:,}, "
          f"{probe.fit_rows:,}행으로 그리고 {probe.stop_rows:,}행으로 판단)")

    # 2단계: 그 수를 고정해 학습셋 전체로 다시 학습한다.
    # 1단계 모델을 그대로 쓰면 떼어낸 62,006행만큼 손해다(평가셋 PR-AUC 0.5468 -> 0.5116).
    print(f"학습셋 전체 {len(X['train']):,}행으로 다시 학습 중...")
    trained = refit_on_all(X["train"], y["train"], trees, model_cfg)

    stem = "xgboost_baseline" if final else "xgboost_val_only"
    model_path = OUT_DIR / f"{stem}.ubj"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kept = trained.save(model_path)

    model_block = {
        "params": dict(trained.params),
        "trees": trained.best_iteration + 1,
        "tree_source": trained.tree_source,
        "fit_rows": trained.fit_rows,
        # 나무 수는 학습셋 뒤쪽 조각으로 정했다. 검증셋은 τ에만 쓴다.
        "trees_chosen_on": {
            "fit_rows": probe.fit_rows,
            "stop_rows": probe.stop_rows,
            "early_stop_ratio": model_cfg["early_stop_ratio"],
        },
        "saved_trees": kept,
    }

    # "얼마나 자주 썼나"와 "총 기여가 얼마나"는 다른 순위를 준다. 둘 다 찍는다.
    # 어느 쪽도 "이 컬럼이 없으면 얼마나 나빠지나"는 아니다 — 그건 순열로만 나온다.
    report(p, trained, model_block, stem, importance=feature_importance(trained, top=15))
    print(f"      {short_path(model_path)}  (나무 {kept:,}그루만 남김)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final",
        action="store_true",
        help="평가셋 수치까지 낸다. 설정을 고르는 동안은 켜지 말 것.",
    )
    main(parser.parse_args().final)
