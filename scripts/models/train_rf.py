"""Random Forest 탐지 모델을 학습하고 공격 전 기준선을 잰다.

실행:
    python scripts/models/train_rf.py            # 검증셋까지만 본다
    python scripts/models/train_rf.py --final    # 평가셋 수치까지 낸다

XGBoost와 같은 데이터·같은 분할·같은 전처리를 쓴다(`src/models/runner.py`). 다른 것은
나무를 쌓는 방식뿐이고, 조기 종료가 없어서 단계가 하나다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.rf import feature_importance, train_rf
from src.models.runner import OUT_DIR, prepare, report, short_path


def main(final: bool) -> None:
    # 조기 종료가 없으므로 학습셋 뒤쪽을 떼지 않는다. 떼면 쓰지도 않을 조각에
    # 학습셋 크기만큼 메모리를 한 번 더 잡는다.
    p = prepare(final=final, stop_split=False)

    forest = p.model_cfg["random_forest"]
    print(f"학습셋 전체 {len(p.X['train']):,}행으로 나무 {forest['n_estimators']:,}그루를 기르는 중...")
    trained = train_rf(p.X["train"], p.y["train"], p.model_cfg)

    stem = "rf_baseline" if final else "rf_val_only"
    model_path = OUT_DIR / f"{stem}.joblib"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    size = trained.save(model_path)

    model_block = {
        "params": dict(trained.params),
        "trees": len(trained.model.estimators_),
        # XGBoost는 조기 종료로 나무 수를 정하지만 숲은 config 값을 그대로 쓴다.
        "tree_source": "fixed",
        "fit_rows": trained.fit_rows,
        "saved_bytes": size,
    }

    # MDI는 값 종류가 많은 컬럼을 과대평가한다. card1(12,242종)이 있어 실제로 걸린다.
    # 순위를 그대로 읽지 말고 참고용으로만 본다(src/models/rf.py 설명 참조).
    report(p, trained, model_block, stem, importance=feature_importance(trained, top=15))
    print(f"      {short_path(model_path)}  ({size / 1e6:,.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final",
        action="store_true",
        help="평가셋 수치까지 낸다. 설정을 고르는 동안은 켜지 말 것.",
    )
    main(parser.parse_args().final)
