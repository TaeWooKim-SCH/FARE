"""세 모델이 공유하는 학습 준비와 결과 보고.

모델마다 다른 것은 '어떻게 학습하나' 하나뿐이다. 데이터를 읽고 시간순으로 자르고 전처리하는
앞부분과, τ를 정하고 지표를 재고 결과를 남기는 뒷부분은 셋이 똑같아야 한다. 같지 않으면
3종 비교가 조건이 다른 숫자를 나란히 놓는 일이 된다(research-plan.md 4.2 ③).

**평가셋 수치는 기본으로 안 낸다.** 설정을 바꿔가며 여러 번 돌리고 그때마다 평가셋 숫자를
보면, 코드에 누수가 없어도 사람 손을 거쳐 평가셋이 선택셋이 된다. 설정을 고르는 동안은
검증셋만 보고, 최종 수치를 뽑을 때 한 번만 `--final`로 연다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.loader import REPO_ROOT, load_config, load_merged
from src.data.split import split_tail, time_split
from src.models.adapter import fit_pass_through
from src.models.metrics import choose_threshold, evaluate_at, rank_metrics
from src.models.preprocess import Preprocessor, fit_preprocessor, target_of

OUT_DIR = REPO_ROOT / "results" / "models"


def short_path(path: Path) -> str:
    """저장소 안이면 상대 경로로 줄이고, 밖이면 그대로 보여준다.

    화면에 찍는 용도라 저장소 밖 경로(테스트의 임시 폴더 같은)에서 예외가 나면 안 된다.
    """
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


@dataclass(frozen=True)
class Prepared:
    """학습 직전 상태. 여기까지는 모델이 무엇이든 똑같다.

    **자른 조각(`Split`)을 통째로 들고 있지 않는다.** 들고 있으면 `--final`이 아닐 때도
    `prepared.split.test`로 평가셋에 닿을 수 있다. 여기 남는 평가셋 흔적은 상품별로
    쪼갤 때 쓰는 `ProductCD` 한 컬럼뿐이고, 그마저 `--final`이 아니면 None이다.
    """

    pre: Preprocessor
    # 트리는 PassThrough, MLP는 MlpAdapter가 들어온다. apply(X) 하나만 지키면 되므로
    # 형을 좁히지 않는다. 좁히면 어댑터를 늘릴 때마다 여기를 고쳐야 한다.
    adapter: object
    X: dict[str, pd.DataFrame]
    y: dict[str, pd.Series]
    data_cfg: dict
    model_cfg: dict
    final: bool
    # 평가셋의 ProductCD. 상품별 기준선을 낼 때만 쓴다(6.1). --final이 아니면 None이다.
    product: pd.Series | None


def prepare(final: bool, stop_split: bool = True, fit_adapter=fit_pass_through) -> Prepared:
    """데이터를 읽어 시간순으로 자르고 전처리까지 끝낸다.

    `stop_split`은 학습셋 뒤쪽을 조기 종료용으로 떼어낼지 정한다. XGBoost는 나무 수를
    정하려고 필요하지만 Random Forest와 MLP는 조기 종료가 없어서 안 쓴다. 안 쓰는 조각을
    굳이 만들면 학습셋 크기만큼 메모리를 한 번 더 잡는다.

    `fit_adapter`는 모델별 입력 변환을 만든다. 기본값은 값을 안 건드리는 트리용이고,
    MLP는 `fit_mlp_adapter`를 넘긴다. **어느 어댑터를 넣든 학습셋만 보고 만든다** —
    아래에서 넘기는 것이 `applied["train"]` 하나뿐이라 다른 조각에 닿을 자리가 없다.
    """
    data_cfg = load_config("config/data.yaml")
    model_cfg = load_config("config/model.yaml")

    print("데이터를 읽는 중...")
    split = time_split(load_merged(data_cfg), data_cfg)

    parts = {"train": split.train, "val": split.val}
    if stop_split:
        # 학습셋을 다시 시간순으로 갈라, 앞쪽으로 나무를 그리고 뒤쪽으로 멈출 시점을 정한다.
        # 검증셋은 τ를 정하는 데만 쓴다.
        fit_part, stop_part = split_tail(split.train, model_cfg["early_stop_ratio"], data_cfg)
        parts = {"fit": fit_part, "stop": stop_part, **parts}
    if final:
        parts["test"] = split.test

    # 규칙은 학습셋에서만 만든다. 나머지에는 적용만 한다.
    pre = fit_preprocessor(split.train)
    applied = {name: pre.apply(part) for name, part in parts.items()}
    adapter = fit_adapter(applied["train"], pre, model_cfg)

    X = {name: adapter.apply(frame) for name, frame in applied.items()}
    y = {name: target_of(part) for name, part in parts.items()}

    print(f"입력 {len(pre.feature_columns)}개 컬럼 / "
          + " ".join(f"{n} {len(v):,}행" for n, v in X.items()))
    if not final:
        print("  (평가셋은 안 읽는다. 최종 수치를 낼 때 --final로 연다)")

    # 상품별로 쪼갤 컬럼만 따로 빼둔다. 평가셋 프레임 자체는 여기서 버린다.
    product = split.test["ProductCD"] if final and "ProductCD" in split.test else None

    return Prepared(
        pre=pre, adapter=adapter, X=X, y=y, product=product,
        data_cfg=data_cfg, model_cfg=model_cfg, final=final,
    )


def report(
    prepared: Prepared,
    trained,
    model_block: dict,
    stem: str,
    importance: pd.DataFrame | None = None,
) -> Path:
    """τ를 정하고 지표를 재서 화면과 파일에 남긴다. 저장한 json 경로를 돌려준다.

    `trained`는 `score(X) -> 확률` 하나만 있으면 된다. 모델 종류를 여기서 알 필요가 없다.
    """
    X, y, beta = prepared.X, prepared.y, prepared.model_cfg["threshold"]["beta"]

    # τ를 먼저 정하고 나서 평가셋 점수를 낸다. 순서만 봐도 평가셋이 τ에 안 닿는 게 보인다.
    val_score = trained.score(X["val"])
    tau = choose_threshold(y["val"], val_score, beta=beta)
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

    by_product = _by_product(prepared, y, score, tau) if prepared.product is not None else []

    if importance is not None:
        print(f"\n=== 상위 {len(importance)}개 컬럼 ===")
        print(importance.to_string(index=False, float_format=lambda v: f"{v:,.4f}"))

    payload = {
        # 평가셋을 봤는지 결과 파일에 남긴다. 나중에 이 숫자를 인용할 때 근거가 된다.
        "test_included": prepared.final,
        "config": {"data": prepared.data_cfg, "model": prepared.model_cfg},
        "preprocess": {
            "feature_count": len(prepared.pre.feature_columns),
            "category_count": len(prepared.pre.category_columns),
            "fit_rows": prepared.pre.fit_rows,
            "fit_time_range": list(prepared.pre.fit_time_range),
        },
        "model": model_block,
        "threshold": {
            "tau": tau.tau, "beta": tau.beta, "chosen_on": "val",
            "val_recall": tau.recall, "val_precision": tau.precision,
        },
        "rank_metrics": ranks,
        "at_threshold": at_tau,
        "by_product": by_product,
        "top_features": importance.to_dict("records") if importance is not None else [],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stem}.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {short_path(out)}")
    return out


def _by_product(prepared: Prepared, y: dict, score: dict, tau) -> list[dict]:
    """평가셋을 상품별로 쪼갠다. W는 D 계열이 통째로 비어 있어 조건이 다르다.

    전체 숫자만 보면 74%를 차지하는 W에 나머지가 묻힌다.
    """
    product = prepared.product
    rows = []
    print("\n=== 평가셋 상품별 ===")
    print(f"{'ProductCD':<11s}{'행':>9s}{'사기율':>9s}{'Recall':>9s}{'Precision':>11s}{'PR-AUC':>9s}")
    for code, idx in product.groupby(product, observed=True).groups.items():
        # product.index는 0부터가 아니라 472432부터다. 라벨을 위치로 바꿔야 점수와 맞물린다.
        pos = product.index.get_indexer(idx)
        yy, ss = y["test"].to_numpy()[pos], score["test"][pos]
        if yy.sum() == 0:
            print(f"{str(code):<11s}{len(yy):>9,}  사기 거래 없음")
            continue
        m = evaluate_at(yy, ss, tau.tau, tau.beta)
        r = rank_metrics(yy, ss)
        rows.append({"ProductCD": str(code), **r, **m})
        print(f"{str(code):<11s}{len(yy):>9,}{yy.mean():>9.4f}{m['recall']:>9.4f}"
              f"{m['precision']:>11.4f}{r['pr_auc']:>9.4f}")
    return rows
