"""하이퍼파라미터 무작위 탐색.

계획서 4.2 ④가 "기본값을 그대로 쓰지 않고 탐색을 거친 뒤 탐색 범위와 최종 설정을 논문에
기재한다"고 요구한다. **목적이 점수를 짜내는 것이 아니라 그 표를 만드는 것**이므로, 언제
멈출지를 미리 못 박는 것이 설계의 핵심이다. 시행 횟수를 `config/search.yaml`에 고정하고
매 시행의 설정과 점수를 전부 남긴다. 그 기록이 그대로 논문 표가 된다.

**학습셋 뒤쪽 조각(stop)에 대고 고른다. 검증셋은 τ 전용으로 계속 비워 둔다.** 검증셋에
대고 30번 고르면 그중 최댓값을 뽑는 셈이라, 실력의 최댓값만이 아니라 잡음의 최댓값까지
같이 뽑는다. 그 잡음이 있는 자리가 경계 근처 행인데 τ를 정하는 것도 같은 행들이다. 이미
같은 일을 약하게 겪었다 — 조기 종료를 검증셋으로 했더니 1,586그루까지 올라갔는데 평가셋은
800그루에서 꺾였다(`src/models/xgb.py`).

**stop 조각은 부풀려져도 된다.** 그 점수는 설정을 고르는 데만 쓰이고 밖으로 안 나간다.
대가는 "진짜 최적 설정을 못 골랐을 수 있다"는 성능 손해지 결과가 무효가 되는 문제가 아니다.

**중간에 끊겨도 이어서 돌 수 있다.** 시행마다 결과를 파일에 이어 쓰고, 다시 돌리면 이미
끝난 시행 수만큼 건너뛴다. 설정은 시드를 고정한 난수로 뽑으므로 n번째 시행은 언제 뽑아도
같은 값이다. 열몇 시간짜리 실행이 죽어서 처음부터 다시 하는 일이 없어야 한다.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.models.metrics import rank_metrics


def sample_params(space: dict, rng: np.random.Generator) -> dict:
    """탐색 공간에서 설정 하나를 뽑는다.

    리스트면 그중 하나를 고르고, `{log: [a, b]}`면 로그 눈금으로, `{uniform: [a, b]}`면
    고르게 뽑는다. 학습률·규제처럼 자릿수가 중요한 값은 로그로 뽑아야 0.001과 0.01 사이가
    0.1과 1 사이만큼 자주 뽑힌다. 고르게 뽑으면 큰 값 쪽만 잔뜩 보게 된다.
    """
    picked = {}
    for name, choices in space.items():
        if isinstance(choices, dict):
            low, high = choices.get("log") or choices["uniform"]
            if "log" in choices:
                picked[name] = float(np.exp(rng.uniform(np.log(low), np.log(high))))
            else:
                picked[name] = float(rng.uniform(low, high))
        else:
            # rng.choice는 리스트 안의 리스트를 배열로 펴버린다. 은닉층 [256, 64]가
            # 그렇게 망가지므로 자리 번호를 뽑아서 원래 값을 그대로 꺼낸다.
            picked[name] = choices[int(rng.integers(len(choices)))]
    return picked


def put(config: dict, path: tuple[str, ...], params: dict) -> dict:
    """config의 지정한 자리에 설정을 얹은 새 config를 만든다. 원본은 안 건드린다."""
    merged = json.loads(json.dumps(config))  # 깊은 복사. yaml에서 온 값이라 전부 기본형이다
    target = merged
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = {**target[path[-1]], **params}
    return merged


def load_done(path: Path) -> list[dict]:
    """이미 끝난 시행을 읽는다. 파일이 없으면 빈 목록이다."""
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def search(
    name: str,
    space: dict,
    search_cfg: dict,
    model_cfg: dict,
    param_path: tuple[str, ...],
    fit_fn,
    X: dict,
    y: dict,
    out_dir: Path,
) -> dict:
    """`trials`번 돌면서 stop 조각 점수가 제일 좋은 설정을 찾는다.

    `fit_fn(X_fit, y_fit, X_stop, y_stop, config) -> 학습된 모델`이면 된다. XGBoost는
    stop 조각을 조기 종료에 쓰고, RF와 MLP는 안 쓴다. 어느 쪽이든 **점수는 stop 조각에서**
    낸다. XGBoost는 멈출 시점을 정한 조각으로 설정도 고르는 셈이라 그만큼 더 부풀려지는데,
    밖으로 안 나가는 숫자라 결과를 무효로 만들지는 않는다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.jsonl"
    done = load_done(path)
    metric = search_cfg["metric"]
    rng = np.random.default_rng(search_cfg["seed"])

    # 건너뛸 시행도 난수는 소비해야 n번째가 언제나 같은 설정이 된다.
    plan = [sample_params(space, rng) for _ in range(search_cfg["trials"])]
    if done:
        print(f"  이미 끝난 시행 {len(done)}개를 건너뛴다")

    for index in range(len(done), len(plan)):
        params = plan[index]
        started = time.perf_counter()
        trained = fit_fn(X["fit"], y["fit"], X["stop"], y["stop"], put(model_cfg, param_path, params))
        score = rank_metrics(y["stop"], trained.score(X["stop"]))[metric]
        record = {
            "trial": index,
            "params": params,
            metric: score,
            "seconds": round(time.perf_counter() - started, 1),
            # 몇 그루/몇 바퀴에서 멈췄는지. 범위 끝에 닿았으면 범위를 잘못 잡은 것이다.
            "rounds": getattr(trained, "rounds", None) or getattr(trained, "best_iteration", None),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        done.append(record)
        best = max(done, key=lambda r: r[metric])
        print(f"  [{index + 1}/{len(plan)}] {metric} {score:.4f} "
              f"({record['seconds']:.0f}초)  최고 {best[metric]:.4f} (시행 {best['trial']})")

    best = max(done, key=lambda r: r[metric])
    return {
        "model": name,
        "space": space,
        "trials_planned": search_cfg["trials"],
        "trials_done": len(done),
        "metric": metric,
        "chosen_on": "stop",
        "best": best,
        "all": done,
    }
