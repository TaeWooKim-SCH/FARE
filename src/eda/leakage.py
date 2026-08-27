"""C·D에 미래가 섞였는지 확인한다.

주최 측이 이 값들을 전체 기간으로 계산했다면 학습 구간에 미래가 녹아 있고, 그러면
모델을 어떻게 학습하든 결과가 무효다. 계산 방법이 공개되지 않아 직접 확인은 못 하므로
간접 신호를 본다.

가장 분명한 신호는 카드의 첫 거래다. 미래까지 셌다면 첫 거래에서도 그 카드의 총 거래
수가 보여야 한다. 실제로 거의 안 보이면 과거만 센다는 뜻이다.
"""

from __future__ import annotations

import pandas as pd

DAY_SECONDS = 86400


def add_uid(df: pd.DataFrame) -> pd.Series:
    """카드를 대신 알아보는 조합. column-reference.md의 정의를 따른다.

    IEEE-CIS에는 카드를 알아볼 번호가 없어서 card1과 addr1, 그리고 D1로 거슬러 올라간
    시점을 붙여 대신 쓴다. 주최 측이 확인해준 것이 아니라 커뮤니티에서 나온 방법이다.
    """
    for column in ("card1", "addr1", "D1", "TransactionDT"):
        if column not in df.columns:
            raise KeyError(f"{column} 컬럼이 있어야 카드를 짚을 수 있습니다.")
    day = df["TransactionDT"] // DAY_SECONDS
    return df["card1"].astype(str) + "_" + df["addr1"].astype(str) + "_" + (day - df["D1"]).astype(str)


def first_row_reveals_total(
    df: pd.DataFrame, columns: list[str], min_transactions: int = 3
) -> pd.DataFrame:
    """카드의 첫 거래에서 이미 그 카드의 총 거래 수가 보이는 비율.

    미래까지 센 값이라면 첫 거래에서도 총 거래 수가 보여야 하므로 이 비율이 1에 가깝다.
    과거만 센 값이라면 첫 거래에서는 셀 과거가 없으므로 비율이 0에 가깝다.
    """
    frame = df.assign(_uid=add_uid(df)).sort_values("TransactionDT", kind="mergesort")
    grouped = frame.groupby("_uid", sort=False)
    frame["_total"] = grouped["TransactionDT"].transform("size")
    frame["_order"] = grouped.cumcount()

    first = frame[(frame["_order"] == 0) & (frame["_total"] >= min_transactions)]
    if first.empty:
        raise ValueError(f"거래 {min_transactions}건 이상인 카드가 없습니다.")

    return pd.DataFrame(
        [
            {
                "column": c,
                "match_rate": float((first[c] == first["_total"]).mean()),
                "cards": len(first),
            }
            for c in columns
        ]
    )


def tail_median_ratio(df: pd.DataFrame, columns: list[str], tail_days: int = 1) -> pd.DataFrame:
    """마지막 며칠의 중앙값이 전체 중앙값과 얼마나 다른가.

    전체 기간으로 셌다면 마지막 날 거래도 앞으로 일어날 일까지 세었을 테니, 중간이
    아니라 끝에서 값이 튄다. 비율이 1에 가까우면 그런 흔적이 없다는 뜻이다.
    """
    day = df["TransactionDT"] // DAY_SECONDS
    last = int(day.max())
    tail = day > last - tail_days

    rows = []
    for c in columns:
        overall = float(df[c].median())
        recent = float(df.loc[tail, c].median())
        rows.append(
            {
                "column": c,
                "median_all": overall,
                "median_tail": recent,
                "ratio": (recent / overall) if overall else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def direction_share(df: pd.DataFrame, columns: list[str], min_transactions: int = 5) -> pd.DataFrame:
    """같은 카드 안에서 값이 늘어나는지 줄어드는지 세어본다.

    과거만 쌓아 세는 값이라면 줄어들 수 없다. 줄어드는 일이 잦다면 다른 방식으로
    계산했거나, 우리가 카드를 잘못 묶어 서로 다른 카드가 섞였다는 뜻이다.
    """
    frame = df.assign(_uid=add_uid(df)).sort_values("TransactionDT", kind="mergesort")
    grouped = frame.groupby("_uid", sort=False)
    frame["_total"] = grouped["TransactionDT"].transform("size")
    enough = frame[frame["_total"] >= min_transactions]

    rows = []
    for c in columns:
        diff = enough.groupby("_uid", sort=False)[c].diff().dropna()
        rows.append(
            {
                "column": c,
                "up": float((diff > 0).mean()),
                "same": float((diff == 0).mean()),
                "down": float((diff < 0).mean()),
            }
        )
    return pd.DataFrame(rows)
