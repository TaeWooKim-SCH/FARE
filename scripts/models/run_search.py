"""세 모델의 하이퍼파라미터를 무작위 탐색한다.

실행:
    python scripts/models/run_search.py                      # 셋 다, config에 적힌 횟수만큼
    python scripts/models/run_search.py --model mlp          # 하나만
    python scripts/models/run_search.py --trials 3           # 짧게 돌려보기

**한 번에 열몇 시간이 걸린다.** 중간에 끊어도 된다 — 시행마다 결과를 파일에 이어 쓰고,
다시 돌리면 끝난 만큼 건너뛴다. 다 못 돌아도 돌아간 만큼으로 고를 수 있고, 몇 번 돌았는지가
결과 파일에 남는다.

**평가셋을 안 읽는다.** 설정은 학습셋 뒤쪽 조각(stop)에서만 고르고, 검증셋은 τ 전용으로
비워 둔다(`src/models/search.py` 설명 참조).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import REPO_ROOT, load_config
from src.models.adapter import fit_pass_through
from src.models.mlp import train_mlp
from src.models.mlp_adapter import fit_mlp_adapter
from src.models.rf import train_rf
from src.models.runner import prepare, short_path
from src.models.search import search
from src.models.xgb import train_xgb

OUT_DIR = REPO_ROOT / "results" / "search"

# 모델마다 (설정을 얹을 자리, 입력 어댑터, 학습 함수).
# 학습 함수는 넷 다 받되 stop 조각을 쓰는 것은 XGBoost뿐이다 — 조기 종료에 쓴다.
# RF와 MLP는 안 쓰고, 점수만 stop에서 낸다.
MODELS = {
    "xgboost": (
        ("xgboost",),
        fit_pass_through,
        lambda X_fit, y_fit, X_stop, y_stop, cfg: train_xgb(X_fit, y_fit, X_stop, y_stop, cfg),
    ),
    "random_forest": (
        ("random_forest",),
        fit_pass_through,
        lambda X_fit, y_fit, X_stop, y_stop, cfg: train_rf(X_fit, y_fit, cfg),
    ),
    "mlp": (
        ("mlp", "network"),
        fit_mlp_adapter,
        lambda X_fit, y_fit, X_stop, y_stop, cfg: train_mlp(X_fit, y_fit, cfg),
    ),
}


def main(names: list[str], trials: int | None) -> None:
    search_cfg = dict(load_config("config/search.yaml"))
    model_cfg = load_config("config/model.yaml")
    if trials is not None:
        search_cfg["trials"] = trials
        print(f"시행 횟수를 {trials}로 덮어쓴다 (연기 테스트용)")

    summaries = {}
    for name in names:
        param_path, fit_adapter, fit_fn = MODELS[name]
        print(f"\n{'=' * 60}\n{name} 탐색 시작\n{'=' * 60}")

        # 모델마다 입력 어댑터가 달라서 준비를 따로 한다. 원본 CSV를 다시 읽는 값을
        # 치르는 대신, 한 번에 한 모델의 입력만 메모리에 있다.
        p = prepare(final=False, stop_split=True, fit_adapter=fit_adapter)
        print(f"  나무를 그릴 {len(p.X['fit']):,}행 / 설정을 고를 {len(p.X['stop']):,}행")

        summaries[name] = search(
            name=name,
            space=search_cfg[name],
            search_cfg=search_cfg,
            model_cfg=p.model_cfg,
            param_path=param_path,
            fit_fn=fit_fn,
            X=p.X,
            y=p.y,
            out_dir=OUT_DIR,
        )
        del p

    print(f"\n{'=' * 60}\n탐색 결과\n{'=' * 60}")
    print(f"{'모델':<16s}{'시행':>7s}{'최고 PR-AUC':>13s}   최종 설정")
    for name, s in summaries.items():
        best = s["best"]
        print(f"{name:<16s}{s['trials_done']:>7d}{best[s['metric']]:>13.4f}   {best['params']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "summary.json"
    payload = {
        "config": {"search": search_cfg, "model": model_cfg},
        # 이 파일 하나로 "무엇을 어디까지 찾아봤나"가 논문에 그대로 들어간다.
        "models": summaries,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: {short_path(out)}")
    print("      최종 설정을 config/model.yaml에 옮겨 적은 뒤 기준선을 다시 낸다")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), action="append",
                        help="이 모델만 돌린다. 여러 번 줄 수 있다. 안 주면 셋 다.")
    parser.add_argument("--trials", type=int,
                        help="config의 시행 횟수를 덮어쓴다. 짧게 돌려볼 때 쓴다.")
    args = parser.parse_args()
    main(args.model or list(MODELS), args.trials)
