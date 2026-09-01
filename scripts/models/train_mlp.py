"""MLP 탐지 모델을 학습하고 공격 전 기준선을 잰다.

실행:
    python scripts/models/train_mlp.py            # 검증셋까지만 본다
    python scripts/models/train_mlp.py --final    # 평가셋 수치까지 낸다

XGBoost·Random Forest와 같은 데이터·같은 분할·같은 임계값 절차를 쓴다(`runner.py`).
다른 것은 입력 변환뿐이라, `prepare`에 어댑터만 갈아 끼운다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.mlp import loss_curve, train_mlp
from src.models.mlp_adapter import fit_mlp_adapter
from src.models.runner import OUT_DIR, prepare, report, short_path


def _adapter_record(adapter) -> dict:
    """어댑터가 무엇을 어떻게 바꿨는지 결과 파일에 남긴다.

    입력 컬럼 수만 적으면 나중에 이 숫자를 다시 만들 수 없다. 원핫 칸이 몇 개였고 결측
    표시가 몇 무리로 묶였는지까지 있어야 같은 입력이 나온다.
    """
    return {
        "onehot_columns": len(adapter.onehot),
        "onehot_slots": sum(len(codes) for codes in adapter.onehot.values()),
        "frequency_columns": sorted(adapter.frequency),
        "missing_groups": len(adapter.missing_groups),
        "missing_columns_grouped": sum(len(group) for group in adapter.missing_groups),
        "quantile_columns": len(adapter.quantile_columns),
        "output_columns": len(adapter.feature_columns),
    }


def main(final: bool) -> None:
    # 조기 종료용 조각을 안 쓴다. sklearn MLP는 eval_set을 못 받아서 몇 바퀴 돌지를
    # 하이퍼파라미터로 두고 다음 PR의 탐색에서 정한다(src/models/mlp.py 설명 참조).
    p = prepare(final=final, stop_split=False, fit_adapter=fit_mlp_adapter)

    record = _adapter_record(p.adapter)
    print(f"MLP 입력 {record['output_columns']}컬럼 "
          f"(원핫 {record['onehot_columns']}개 컬럼 -> 칸 {record['onehot_slots']}개, "
          f"빈도 {len(record['frequency_columns'])}개, "
          f"결측 표시 {record['missing_columns_grouped']}개 -> 무리 {record['missing_groups']}개)")

    network = p.model_cfg["mlp"]["network"]
    print(f"학습셋 전체 {len(p.X['train']):,}행으로 학습 중... "
          f"(은닉층 {network['hidden_layer_sizes']}, 최대 {network['max_iter']}바퀴)")
    trained = train_mlp(p.X["train"], p.y["train"], p.model_cfg)

    멈춘이유 = "손실이 평평해짐" if trained.stopped_early else "max_iter에 닿음"
    print(f"  {trained.rounds}바퀴 돌고 멈춤 ({멈춘이유}). 마지막 학습 손실 {trained.final_loss:.5f}")

    stem = "mlp_baseline" if final else "mlp_val_only"
    model_path = OUT_DIR / f"{stem}.joblib"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    size = trained.save(model_path)

    model_block = {
        "params": dict(trained.params),
        "rounds": trained.rounds,
        # 왜 멈췄는지가 뜻이 다르다. max_iter에 닿았으면 더 돌 여지가 남은 것이다.
        "round_source": "loss_plateau" if trained.stopped_early else "max_iter",
        "final_loss": trained.final_loss,
        "fit_rows": trained.fit_rows,
        "adapter": record,
        "loss_curve": loss_curve(trained)["loss"].tolist(),
        "saved_bytes": size,
    }

    # 신경망에는 트리의 "어느 컬럼을 몇 번 썼나"에 해당하는 값이 없다. 컬럼 기여도는
    # 순열 중요도로만 나오는데 543컬럼마다 다시 예측해야 해서 비싸다. 여기서는 안 낸다.
    report(p, trained, model_block, stem, importance=None)
    print(f"      {short_path(model_path)}  ({size / 1e6:,.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--final",
        action="store_true",
        help="평가셋 수치까지 낸다. 설정을 고르는 동안은 켜지 말 것.",
    )
    main(parser.parse_args().final)
