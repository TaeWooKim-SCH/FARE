"""IEEE-CIS 데이터 로딩.

BAF·Elliptic으로 확장할 때 이 모듈만 갈아끼우면 나머지 파이프라인을 그대로 쓴다
(research-plan.md 4.4).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "data.yaml"


def load_config(path: str | Path | None = None) -> dict:
    """설정 파일을 읽는다. 시드·분할 비율·경로가 전부 여기서 나온다."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(cfg: dict, key: str) -> Path:
    paths = cfg["paths"]
    return REPO_ROOT / paths["root"] / paths[key]


def _read(path: Path, columns: Iterable[str] | None, nrows: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. datasets/README.md 를 보고 Kaggle에서 받아 배치하세요."
        )
    wanted = list(columns) if columns else None
    # 683MB를 통째로 읽으면 2.2GB를 먹는다. 필요한 컬럼만 읽는 경로를 열어둔다.
    frame = pd.read_csv(path, usecols=wanted, nrows=nrows, low_memory=False)
    if wanted:
        # usecols는 파일에 적힌 순서로 돌려준다. 부르는 쪽이 정한 순서를 지켜야
        # 나중에 특징 행렬을 만들 때 열이 조용히 뒤바뀌지 않는다.
        frame = frame[wanted]
    return frame


def load_transactions(
    cfg: dict | None = None,
    columns: Iterable[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """거래 테이블(590,540행 x 394열)을 읽는다."""
    cfg = cfg or load_config()
    return _read(_resolve(cfg, "transaction"), columns, nrows)


def load_identity(
    cfg: dict | None = None,
    columns: Iterable[str] | None = None,
    nrows: int | None = None,
) -> pd.DataFrame:
    """기기·네트워크 지문 테이블(144,233행 x 41열)을 읽는다."""
    cfg = cfg or load_config()
    return _read(_resolve(cfg, "identity"), columns, nrows)


def load_merged(cfg: dict | None = None, nrows: int | None = None) -> pd.DataFrame:
    """거래에 identity를 왼쪽 조인한다.

    identity는 거래의 24.4%에만 있다. 없는 쪽을 버리면 데이터의 3/4가 날아가므로
    왼쪽 조인으로 결측을 남긴다. 결측 처리 방식은 모델 단계에서 정한다
    (feature-taxonomy.md 6절).
    """
    cfg = cfg or load_config()
    tx = load_transactions(cfg, nrows=nrows)
    idf = load_identity(cfg)
    merged = tx.merge(idf, on=cfg["id_column"], how="left")
    if len(merged) != len(tx):
        raise ValueError(
            f"조인 후 행 수가 변했습니다: {len(tx):,} -> {len(merged):,}. "
            f"{cfg['id_column']}가 identity에서 중복인지 확인하세요."
        )
    return merged
