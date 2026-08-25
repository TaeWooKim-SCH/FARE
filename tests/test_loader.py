"""실제 IEEE-CIS 파일에 대고 로더와 분할을 확인한다.

683MB를 읽으므로 느리다. 빼고 돌리려면 `pytest -m "not slow"`.
데이터가 없는 환경에서는 자동으로 건너뛴다.
"""

import pandas as pd
import pytest

from src.data.loader import REPO_ROOT, load_config, load_merged, load_transactions
from src.data.split import time_split

pytestmark = pytest.mark.slow


def data_available() -> bool:
    cfg = load_config()
    root = REPO_ROOT / cfg["paths"]["root"]
    return (root / cfg["paths"]["transaction"]).exists() and (root / cfg["paths"]["identity"]).exists()


needs_data = pytest.mark.skipif(not data_available(), reason="datasets/ieee-cis 에 원본 파일이 없다")


@needs_data
def test_설정_파일에서_시드와_비율을_읽어온다():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["split"]["time_column"] == "TransactionDT"


@needs_data
def test_거래_테이블은_59만행_394열이다():
    tx = load_transactions(columns=None, nrows=None)
    assert tx.shape == (590_540, 394)


@needs_data
def test_사기율은_3점5퍼센트다():
    tx = load_transactions(columns=["isFraud"])
    assert tx["isFraud"].mean() == pytest.approx(0.03499, abs=1e-4)


@needs_data
def test_필요한_컬럼만_읽을_수_있다():
    tx = load_transactions(columns=["TransactionID", "TransactionDT", "isFraud"])
    assert list(tx.columns) == ["TransactionID", "TransactionDT", "isFraud"]


@needs_data
def test_identity를_붙여도_거래_행_수가_변하지_않는다():
    merged = load_merged()
    assert len(merged) == 590_540


@needs_data
def test_identity가_붙는_거래는_전체의_24퍼센트다():
    merged = load_merged()
    assert merged["DeviceType"].notna().mean() == pytest.approx(0.238, abs=0.01)


@needs_data
def test_실제_데이터에서도_학습셋이_평가셋보다_시간상_앞선다():
    tx = load_transactions(columns=["TransactionDT", "isFraud"])
    split = time_split(tx)
    assert split.train["TransactionDT"].max() < split.val["TransactionDT"].min()
    assert split.val["TransactionDT"].max() < split.test["TransactionDT"].min()
    assert len(split) == len(tx)


@needs_data
def test_평가셋에도_사기가_충분히_들어간다():
    # 시간순으로 자르면 사기가 한쪽에 쏠릴 수 있다. 쏠리면 평가가 성립하지 않는다.
    tx = load_transactions(columns=["TransactionDT", "isFraud"])
    split = time_split(tx)
    assert split.val["isFraud"].sum() > 0
    assert split.test["isFraud"].sum() > 0
