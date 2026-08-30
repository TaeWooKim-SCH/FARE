"""XGBoost 탐지 모델을 학습하고 공격 전 기준선을 잰다.

실행:
    python scripts/models/train_xgb.py            # 검증셋까지만 본다
    python scripts/models/train_xgb.py --final    # 평가셋 수치까지 낸다

**평가셋 수치는 기본으로 안 낸다.** 설정을 바꿔가며 여러 번 돌리고 그때마다 평가셋 숫자를
보면, 코드에 누수가 없어도 사람 손을 거쳐 평가셋이 선택셋이 된다. 설정을 고르는 동안은
검증셋만 보고, 최종 수치를 뽑을 때 한 번만 --final로 연다.

결과는 results/models/ 아래에 설정과 함께 남긴다. 설정 없이 숫자만 남기면 나중에
어떤 조건에서 나온 값인지 알 수 없어 재현이 안 된다.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.data.loader import REPO_ROOT, load_config, load_merged
from src.data.split import time_split
from src.models.metrics import choose_threshold, evaluate_at, rank_metrics
from src.models.preprocess import fit_preprocessor, target_of
from src.models.xgb import feature_importance, train_xgb

OUT_DIR = REPO_ROOT / "results" / "models"


def main(final: bool) -> None:
    data_cfg = load_config("config/data.yaml")
    model_cfg = load_config("config/model.yaml")

    print("데이터를 읽는 중...")
    frame = load_merged(data_cfg)
    split = time_split(frame, data_cfg)

    parts = {"train": split.train, "val": split.val}
    if final:
        parts["test"] = split.test

    # 규칙은 학습셋에서만 만든다. 검증셋·평가셋에는 적용만 한다.
    pre = fit_preprocessor(split.train)
    X = {name: pre.apply(part) for name, part in parts.items()}
    y = {name: target_of(part) for name, part in parts.items()}
    print(f"입력 {len(pre.feature_columns)}개 컬럼 / "
          + " ".join(f"{n} {len(v):,}행" for n, v in X.items()))
    if not final:
        print("  (평가셋은 안 읽는다. 최종 수치를 낼 때 --final로 연다)")

    print("학습 중...")
    trained = train_xgb(X["train"], y["train"], X["val"], y["val"], model_cfg)
    print(f"  나무 {trained.best_iteration + 1}그루에서 멈춤 "
          f"(최대 {model_cfg['xgboost']['n_estimators']})")

    # τ를 먼저 정하고 나서 평가셋 점수를 낸다. 순서만 봐도 평가셋이 τ에 안 닿는 게 보인다.
    val_score = trained.score(X["val"])
    tau = choose_threshold(y["val"], val_score, beta=model_cfg["threshold"]["beta"])
    print(f"\n운영 임계값 τ = {tau.tau:.6f}  (검증셋 F{tau.beta:g} 최대)")
    print(f"  검증셋에서 Recall {tau.recall:.4f} / Precision {tau.precision:.4f} / "
          f"표시 비율 {tau.flag_rate:.4%}")

    score = {"val": val_score}
    score.update({n: trained.score(X[n]) for n in X if n != "val"})
    names = [n for n in ("train", "val", "test") if n in X]

    ranks = {n: rank_metrics(y[n], score[n]) for n in names}
    at_tau = {n: evaluate_at(y[n], score[n], tau.tau, tau.beta) for n in names}

    print("\n=== 임계값과 무관한 지표 ===")
    print(f"{'':6s}{'ROC-AUC':>10s}{'PR-AUC':>10s}{'사기율':>10s}")
    for name in names:
        r = ranks[name]
        print(f"{name:6s}{r['roc_auc']:>10.4f}{r['pr_auc']:>10.4f}{r['positive_rate']:>10.4f}")

    print(f"\n=== τ={tau.tau:.6f}에서 ===")
    print(f"{'':6s}{'Recall':>9s}{'Precision':>11s}{'F2':>8s}{'표시비율':>10s}{'놓친 사기':>10s}")
    for name in [n for n in names if n != "train"]:
        m = at_tau[name]
        print(f"{name:6s}{m['recall']:>9.4f}{m['precision']:>11.4f}"
              f"{m[f'f{tau.beta:g}']:>8.4f}{m['flag_rate']:>10.4%}{m['fn']:>10,}")

    # 상품별로 쪼갠다. W는 D 계열이 통째로 비어 있어 조건이 다르다.
    by_product = []
    if final:
        print("\n=== 평가셋 상품별 ===")
        print(f"{'ProductCD':<11s}{'행':>9s}{'사기율':>9s}{'Recall':>9s}"
              f"{'Precision':>11s}{'PR-AUC':>9s}")
        for code, idx in split.test.groupby("ProductCD", observed=True).groups.items():
            # test.index는 0부터가 아니라 472432부터다. 라벨을 위치로 바꿔야 점수와 맞물린다.
            pos = split.test.index.get_indexer(idx)
            yy, ss = y["test"].to_numpy()[pos], score["test"][pos]
            if yy.sum() == 0:
                print(f"{str(code):<11s}{len(yy):>9,}  사기 거래 없음")
                continue
            m = evaluate_at(yy, ss, tau.tau, tau.beta)
            r = rank_metrics(yy, ss)
            by_product.append({"ProductCD": str(code), **r, **m})
            print(f"{str(code):<11s}{len(yy):>9,}{yy.mean():>9.4f}{m['recall']:>9.4f}"
                  f"{m['precision']:>11.4f}{r['pr_auc']:>9.4f}")

    # "얼마나 자주 썼나"와 "총 기여가 얼마나"는 다른 순위를 준다. 둘 다 찍는다.
    # 어느 쪽도 "이 컬럼이 없으면 얼마나 나빠지나"는 아니다 — 그건 순열로만 나온다.
    imp = feature_importance(trained, top=15)
    print("\n=== 총 기여 상위 15개 ===")
    print(f"  {'컬럼':<20s}{'총 기여':>9s}{'분기 비중':>10s}{'분기 수':>9s}")
    for _, row in imp.iterrows():
        print(f"  {row['feature']:<20s}{row['total_gain_share']:>9.2%}"
              f"{row['splits_share']:>10.2%}{row['splits']:>9,}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        # 평가셋을 봤는지 결과 파일에 남긴다. 나중에 이 숫자를 인용할 때 근거가 된다.
        "test_included": final,
        "config": {"data": data_cfg, "model": model_cfg},
        "preprocess": {
            "feature_count": len(pre.feature_columns),
            "category_count": len(pre.category_columns),
            "fit_rows": pre.fit_rows,
            "fit_time_range": list(pre.fit_time_range),
        },
        "model": {
            "params": {k: v for k, v in trained.params.items()},
            "best_iteration": trained.best_iteration,
            "train_rows": trained.train_rows,
            "val_rows": trained.val_rows,
        },
        "threshold": {
            "tau": tau.tau, "beta": tau.beta, "chosen_on": "val",
            "val_recall": tau.recall, "val_precision": tau.precision,
        },
        "rank_metrics": ranks,
        "at_threshold": at_tau,
        "by_product": by_product,
        "top_features": imp.to_dict("records"),
    }
    stem = "xgboost_baseline" if final else "xgboost_val_only"
    model_path = OUT_DIR / f"{stem}.ubj"
    kept = trained.save(model_path)
    payload["model"]["saved_trees"] = kept

    out = OUT_DIR / f"{stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {out.relative_to(REPO_ROOT)}")
    print(f"      {model_path.relative_to(REPO_ROOT)}  (나무 {kept:,}그루만 남김)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final",
        action="store_true",
        help="평가셋 수치까지 낸다. 설정을 고르는 동안은 켜지 말 것.",
    )
    main(parser.parse_args().final)
