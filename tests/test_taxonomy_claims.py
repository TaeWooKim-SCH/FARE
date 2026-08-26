"""feature-taxonomy.md 부록의 핵심 주장을 실제 데이터로 검증한다.

문서에 적힌 수치가 `config/constraints.yaml`과 공격 코드로 흘러가므로, 틀리면 아래로
전부 번진다. 이 테스트가 깨지면 문서가 틀렸거나 데이터가 바뀐 것이고, 둘 다 조용히
넘어가면 안 된다.

여기 담은 것은 다섯 가지다. 틀렸을 때 무언가 실제로 부서지는 주장만 골랐다.
  1. D8·D9가 거래 시각에서 파생된다 — 공격이 시각을 바꿀 때 함께 갱신할 대상
  2. V 결측 패턴 블록의 구성 — constraints.yaml의 기술 단위
  3. ProductCD가 feature 존재를 가른다 — 결과를 상품별로 쪼개 보고할 근거
  4. 금액 형식 제약 — 제약 검증 함수가 정상 거래를 튕겨내지 않으려면 정확해야 한다
  5. D 음수 27건이 정상이다 — 검증 함수가 이 행을 버리면 안 된다

683MB를 읽으므로 느리다. 빼려면 `pytest -m "not slow"`.
"""

import re

import numpy as np
import pandas as pd
import pytest

from src.data.loader import REPO_ROOT, load_config, load_transactions
from src.eda.profile import missing_pattern_blocks, productcd_gating

pytestmark = pytest.mark.slow

D_COLUMNS = [f"D{i}" for i in range(1, 16)]
M_COLUMNS = [f"M{i}" for i in range(1, 10)]


def data_available() -> bool:
    cfg = load_config()
    root = REPO_ROOT / cfg["paths"]["root"]
    return (root / cfg["paths"]["transaction"]).exists()


needs_data = pytest.mark.skipif(not data_available(), reason="datasets/ieee-cis 에 원본 파일이 없다")


@pytest.fixture(scope="module")
def tx() -> pd.DataFrame:
    """거래 테이블 전체. 모듈에서 한 번만 읽는다(2.2GB)."""
    return load_transactions()


@pytest.fixture(scope="module")
def kaggle_test() -> pd.DataFrame:
    """대회 test 파일. 항등식이 학습 구간 밖에서도 성립하는지 확인하는 데 쓴다."""
    cfg = load_config()
    path = REPO_ROOT / cfg["paths"]["root"] / "test_transaction.csv"
    if not path.exists():
        pytest.skip("test_transaction.csv 가 없다")
    return pd.read_csv(path, usecols=["TransactionDT", "TransactionAmt", "ProductCD", "D8", "D9"], low_memory=False)


def hour_of(dt: pd.Series) -> pd.Series:
    return (dt // 3600) % 24


# ── 1. D8·D9는 거래 시각에서 파생된다 ──────────────────────────────────────


@needs_data
def test_D9는_그_거래의_시를_24로_나눈_값이다(tx):
    # 공격이 시각을 바꾸면 D9도 반드시 갱신해야 한다. 안 하면 오탐 0인 검사에 걸린다.
    m = tx["D9"].notna()
    assert m.sum() == 74_926
    mismatch = (np.round(tx.loc[m, "D9"] * 24).astype(int) != hour_of(tx.loc[m, "TransactionDT"])).sum()
    assert mismatch == 0


@needs_data
def test_D9_항등식은_대회_test에서도_성립한다(kaggle_test):
    m = kaggle_test["D9"].notna()
    mismatch = (
        np.round(kaggle_test.loc[m, "D9"] * 24).astype(int) != hour_of(kaggle_test.loc[m, "TransactionDT"])
    ).sum()
    assert mismatch == 0


@needs_data
def test_D8의_소수부도_같은_시를_가리킨다(tx):
    m = tx["D8"].notna()
    mismatch = (np.round((tx.loc[m, "D8"] % 1) * 24).astype(int) != hour_of(tx.loc[m, "TransactionDT"])).sum()
    assert mismatch == 0


@needs_data
def test_D8과_D9는_결측_위치가_완전히_같다(tx):
    # 그래서 D8 검사는 D9와 독립된 두 번째 검사가 아니라 같은 제약의 재진술이다.
    assert (tx["D8"].isna() == tx["D9"].isna()).all()


@needs_data
def test_D9의_고유값은_정확히_24개다(tx):
    assert tx["D9"].dropna().nunique() == 24


@needs_data
def test_D9는_전체의_12점7퍼센트에만_있다(tx):
    # 이 검사가 W(전체의 74%)에서는 발동조차 하지 않는다는 뜻이다.
    assert tx["D9"].notna().mean() == pytest.approx(0.12688, abs=1e-5)


@needs_data
def test_D9는_ProductCD가_W인_거래에_아예_없다(tx):
    assert tx.loc[tx["ProductCD"] == "W", "D9"].notna().sum() == 0


# ── 2. V 결측 패턴 블록 ────────────────────────────────────────────────────


@needs_data
def test_V의_결측_패턴은_정확히_15종이다(tx):
    assert len(missing_pattern_blocks(tx)) == 15


@needs_data
def test_번호가_끊기는_V블록이_8개다(tx):
    # 부록의 "네 구간이 각각 두 블록으로 쪼개져"와 같은 말이다.
    # 끊기는 블록이 있으므로 constraints.yaml에 번호 구간으로 적을 수 없다.
    blocks = missing_pattern_blocks(tx)
    assert int((~blocks["contiguous"]).sum()) == 8


@needs_data
def test_V1부터_V11은_한_블록이고_D11과_결측_위치가_같다(tx):
    blocks = missing_pattern_blocks(tx)
    v1_block = blocks[blocks["columns"].str.startswith("V1,")].iloc[0]
    assert v1_block["n_columns"] == 11
    assert v1_block["columns"] == ",".join(f"V{i}" for i in range(1, 12))
    assert (tx["V1"].isna() == tx["D11"].isna()).all()


# ── 3. ProductCD가 feature 존재를 가른다 ───────────────────────────────────


@pytest.fixture(scope="module")
def gating(tx) -> dict[str, set[str]]:
    table = productcd_gating(tx)
    return {r["product"]: set(filter(None, r["columns"].split(","))) for _, r in table.iterrows()}


@needs_data
def test_W에는_D6부터_D9와_D12부터_D14가_없다(gating):
    assert {"D6", "D7", "D8", "D9", "D12", "D13", "D14"} <= gating["W"]


@needs_data
def test_W에서_전무한_V는_159개다(gating):
    v_gone = [c for c in gating["W"] if re.fullmatch(r"V\d+", c)]
    assert len(v_gone) == 159


@needs_data
def test_dist1은_ProductCD가_W인_거래에만_있다(gating):
    for product in ("C", "R", "H", "S"):
        assert "dist1" in gating[product]
    assert "dist1" not in gating["W"]


@needs_data
def test_조건부_가능_레버가_일부_상품에_아예_없다(gating):
    # 2차로 이메일을 열어도 이 상품에서는 바꿀 값 자체가 없다.
    assert "P_emaildomain" in gating["S"]
    assert "R_emaildomain" in gating["W"]


@needs_data
def test_M은_ProductCD가_W인_거래에만_있다(gating):
    for product in ("R", "H", "S"):
        assert set(M_COLUMNS) <= gating[product]
    # C만 예외로 M4가 살아 있다.
    assert set(M_COLUMNS) - {"M4"} <= gating["C"]
    assert "M4" not in gating["C"]


# ── 4. 금액 형식 제약 ──────────────────────────────────────────────────────


def non_integer_rate(frame: pd.DataFrame, product: str) -> float:
    amounts = frame.loc[frame["ProductCD"] == product, "TransactionAmt"]
    return float((amounts != amounts.round()).mean())


@needs_data
@pytest.mark.parametrize("product", ["H", "R"])
def test_H와_R에는_비정수_금액이_한_건도_없다(tx, kaggle_test, product):
    # 이 상품에 소수 금액을 넣으면 오탐률 0%로 걸린다. 공격이 지켜야 할 제약이다.
    assert non_integer_rate(tx, product) == 0.0
    assert non_integer_rate(kaggle_test, product) == 0.0


@needs_data
def test_C의_정수_금액은_35건뿐이다(tx):
    amounts = tx.loc[tx["ProductCD"] == "C", "TransactionAmt"]
    assert int((amounts == amounts.round()).sum()) == 35


@needs_data
def test_소수_3자리_금액은_사실상_전부_C다(tx):
    decimals = tx["TransactionAmt"].map(lambda v: len(str(float(v)).split(".")[1].rstrip("0")))
    three = tx.loc[decimals >= 3]
    assert len(three) == 61_933
    assert (three["ProductCD"] == "C").mean() > 0.9999


@needs_data
def test_W_금액의_대부분은_0점05의_배수다(tx):
    amounts = tx.loc[tx["ProductCD"] == "W", "TransactionAmt"]
    on_grid = (np.round(amounts * 20) - amounts * 20).abs() < 1e-9
    assert on_grid.mean() == pytest.approx(0.9457, abs=5e-4)


@needs_data
def test_금액은_모두_양수다(tx):
    assert (tx["TransactionAmt"] > 0).all()


# ── 5. D 음수는 정상 데이터다 ──────────────────────────────────────────────


@needs_data
def test_D가_음수인_행은_27건이고_전부_정상_거래다(tx):
    # 제약 검증 함수가 D를 0 이상으로 강제하면 이 27건을 잘못 버린다.
    negative = (tx[D_COLUMNS] < 0).any(axis=1)
    assert int(negative.sum()) == 27
    assert int(tx.loc[negative, "isFraud"].sum()) == 0


@needs_data
def test_D1은_음수가_없고_절반_가까이가_0이다(tx):
    # 그래서 시각을 하루만 앞으로 당겨도 절반이 데이터에 없는 음수가 된다.
    d1 = tx["D1"].dropna()
    assert (d1 >= 0).all()
    assert (d1 == 0).mean() == pytest.approx(0.4754, abs=1e-3)


@needs_data
def test_D2는_항상_D1보다_작거나_같다(tx):
    # D1만 밀고 D2를 고정하면 데이터에 한 건도 없는 상태가 만들어진다.
    both = tx[["D1", "D2"]].dropna()
    assert (both["D1"] >= both["D2"]).all()
