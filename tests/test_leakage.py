"""C·D에 미래가 섞였는지 보는 검사 자체를 검증한다.

검사가 잘못 세면 "미래 오염 없음"이라는 결론이 근거를 잃는다. 세는 규칙을 먼저 고정하고,
느린 테스트로 실제 데이터의 수치가 문서와 맞는지 확인한다.
"""

import pandas as pd
import pytest

from src.data.loader import REPO_ROOT, load_config, load_transactions
from src.eda.leakage import DAY_SECONDS, add_uid, direction_share, first_row_reveals_total, tail_median_ratio

COUNT_COLUMNS = [f"C{i}" for i in range(1, 15)]


def make_card(n_rows: int, counts: list[float], card: int = 1) -> pd.DataFrame:
    """거래가 하루 간격으로 이어지는 카드 하나를 만든다.

    카드를 짚는 조합이 `거래일 - D1`을 쓰므로, D1을 거래일과 함께 늘려야 모든 행이
    같은 카드로 묶인다. D1을 0으로 고정하면 행마다 다른 카드가 되어버린다.
    """
    days = [i + 1 for i in range(n_rows)]
    return pd.DataFrame(
        {
            "TransactionDT": [d * DAY_SECONDS for d in days],
            "card1": card,
            "addr1": 100,
            "D1": [float(d) for d in days],
            "C1": counts,
        }
    )


def test_카드를_짚는_조합에_필요한_컬럼이_없으면_거부한다():
    with pytest.raises(KeyError):
        add_uid(pd.DataFrame({"card1": [1]}))


def test_미래까지_센_값은_첫_거래에서_총_거래수가_보인다():
    # 거래 4건짜리 카드인데 첫 행부터 4가 적혀 있으면 미래를 본 것이다.
    frame = make_card(4, [4.0, 4.0, 4.0, 4.0])
    result = first_row_reveals_total(frame, ["C1"])
    assert result.loc[0, "match_rate"] == 1.0


def test_과거만_센_값은_첫_거래에서_총_거래수가_안_보인다():
    # 1부터 쌓아 세면 첫 행은 1이고 총 거래 수 4와 다르다.
    frame = make_card(4, [1.0, 2.0, 3.0, 4.0])
    result = first_row_reveals_total(frame, ["C1"])
    assert result.loc[0, "match_rate"] == 0.0


def test_거래가_적은_카드는_세지_않는다():
    # 거래 2건짜리 카드는 첫 행이 우연히 2일 수 있어 판단에 넣지 않는다.
    frame = make_card(2, [2.0, 2.0])
    with pytest.raises(ValueError, match="거래 3건 이상"):
        first_row_reveals_total(frame, ["C1"], min_transactions=3)


def test_마지막_날_값이_전체와_같으면_비율이_1이다():
    frame = make_card(10, [5.0] * 10)
    result = tail_median_ratio(frame, ["C1"])
    assert result.loc[0, "ratio"] == pytest.approx(1.0)


def test_마지막_날에_값이_튀면_비율로_드러난다():
    frame = make_card(10, [1.0] * 9 + [100.0])
    result = tail_median_ratio(frame, ["C1"])
    assert result.loc[0, "ratio"] > 10


def test_쌓아_세는_값은_줄어들지_않는다():
    frame = make_card(6, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = direction_share(frame, ["C1"])
    assert result.loc[0, "up"] == 1.0
    assert result.loc[0, "down"] == 0.0


def test_값이_줄어들면_그대로_잡힌다():
    frame = make_card(6, [5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    result = direction_share(frame, ["C1"])
    assert result.loc[0, "down"] == 1.0


# ── 실제 데이터 ────────────────────────────────────────────────────────────


def data_available() -> bool:
    cfg = load_config()
    return (REPO_ROOT / cfg["paths"]["root"] / cfg["paths"]["transaction"]).exists()


@pytest.mark.slow
@pytest.mark.skipif(not data_available(), reason="datasets/ieee-cis 에 원본 파일이 없다")
def test_C에_미래가_섞인_흔적이_없다():
    """이 값이 크게 오르면 C가 미래를 본다는 뜻이고, 그러면 실험 전체가 무효다."""
    tx = load_transactions(columns=["TransactionDT", "card1", "addr1", "D1"] + COUNT_COLUMNS)
    result = first_row_reveals_total(tx, ["C1", "C13", "C14"]).set_index("column")

    assert result.loc["C1", "match_rate"] == pytest.approx(0.03968, abs=1e-4)
    assert result.loc["C13", "match_rate"] == pytest.approx(0.03388, abs=1e-4)
    assert result.loc["C14", "match_rate"] == pytest.approx(0.03447, abs=1e-4)
    # 어느 컬럼도 절반을 넘지 않는다. 넘으면 미래를 보고 있다는 신호다.
    assert first_row_reveals_total(tx, COUNT_COLUMNS)["match_rate"].max() < 0.5
