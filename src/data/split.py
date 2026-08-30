"""시간순 분할.

무작위 분할은 쓰지 않는다. 사기는 한 카드가 짧은 기간에 여러 번 털리는 식으로 뭉쳐서
나오므로, 섞으면 같은 뭉치가 학습셋과 평가셋에 갈려 들어가 모델이 카드 조합을 외운다.
실제 배포는 과거로 학습해 미래를 예측하므로 평가도 그렇게 잘라야 한다
(research-plan.md 9장).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.loader import load_config


@dataclass(frozen=True)
class Split:
    """시간순으로 자른 세 조각. val은 운영 임계값 τ를 정하는 데 쓴다."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def __len__(self) -> int:
        return len(self.train) + len(self.val) + len(self.test)


def _advance_past_ties(sorted_values: pd.Series, index: int) -> int:
    """경계가 같은 시각 한가운데 떨어지면 그 시각이 끝나는 곳까지 민다.

    전체의 5.75%가 같은 초를 공유해서 실제로 일어난다. 밀지 않으면 같은 초의 거래가
    양쪽으로 갈려 train의 마지막 시각과 test의 첫 시각이 같아진다.
    """
    n = len(sorted_values)
    if index <= 0 or index >= n:
        return index
    boundary = sorted_values.iat[index - 1]
    while index < n and sorted_values.iat[index] == boundary:
        index += 1
    return index


def time_split(df: pd.DataFrame, cfg: dict | None = None) -> Split:
    """`TransactionDT` 기준으로 학습·검증·평가를 시간순으로 자른다."""
    cfg = cfg or load_config()
    conf = cfg["split"]
    time_column = conf["time_column"]

    if time_column not in df.columns:
        raise KeyError(f"{time_column} 컬럼이 없습니다. 분할 기준 시각이 있어야 합니다.")
    if df[time_column].isna().any():
        raise ValueError(f"{time_column}에 결측이 있습니다. 시간순 분할의 기준이 될 수 없습니다.")
    if not pd.api.types.is_numeric_dtype(df[time_column]):
        # 문자열이면 사전순으로 정렬돼 '1000'이 '2'보다 앞선다. 아래 겹침 검사도
        # 문자열 비교라 그냥 통과해서, 미래 거래가 학습셋에 들어가도 조용하다.
        raise TypeError(
            f"{time_column}이 숫자가 아닙니다({df[time_column].dtype}). "
            "사전순 정렬이 되면 분할이 뒤집힙니다."
        )

    train_ratio = conf["train_ratio"]
    val_ratio = conf["val_ratio"]
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError(
            f"비율이 잘못됐습니다: train {train_ratio}, val {val_ratio}. "
            "운영 임계값 τ를 정할 검증셋과 평가셋이 모두 남아야 하므로, "
            "둘 다 0보다 크고 합이 1보다 작아야 합니다."
        )

    # mergesort는 안정 정렬이라 같은 시각 거래의 원래 순서가 보존된다.
    ordered = df.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    times = ordered[time_column]
    n = len(ordered)

    # 비율을 더한 뒤 곱하면 0.7 + 0.1이 0.7999...가 되어 한 행씩 밀린다.
    # 조각마다 따로 세서 부동소수점 오차가 쌓이지 않게 한다.
    cut_train = int(n * train_ratio)
    cut_val = min(cut_train + int(n * val_ratio), n)
    if conf.get("keep_same_time_together", True):
        cut_train = _advance_past_ties(times, cut_train)
        cut_val = _advance_past_ties(times, max(cut_val, cut_train))

    split = Split(
        train=ordered.iloc[:cut_train].copy(),
        val=ordered.iloc[cut_train:cut_val].copy(),
        test=ordered.iloc[cut_val:].copy(),
    )
    _assert_ordered(split, time_column)
    return split


def split_tail(
    frame: pd.DataFrame, ratio: float, cfg: dict | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """시간순으로 뒤쪽 `ratio`만큼을 떼어낸다. 앞쪽과 뒤쪽을 돌려준다.

    학습셋에서 조기 종료용 조각을 만드는 데 쓴다. **검증셋은 운영 임계값 τ를 정하는 데만
    써야 한다.** 조기 종료까지 검증셋으로 하면, τ를 정하는 데이터가 이미 모델을 고르는 데
    쓰인 셈이 되어 검증셋 숫자가 부풀려진다. 첫 실행에서 나무 수를 잘라가며 재보니 평가셋
    PR-AUC는 800그루에서 꼭대기를 찍고 내려가는데 검증셋은 끝까지 올라, 1,586그루까지
    갔다. 그 사이에 벌어진 만큼이 부풀려진 몫이다.

    떼어낸 조각도 시간순으로 뒤쪽이다. 무작위로 뽑으면 미래가 학습 쪽으로 샌다.
    """
    cfg = cfg or load_config()
    conf = cfg["split"]
    time_column = conf["time_column"]

    if not 0 < ratio < 1:
        raise ValueError(f"비율이 잘못됐습니다: {ratio}. 0보다 크고 1보다 작아야 합니다.")
    if time_column not in frame.columns:
        raise KeyError(f"{time_column} 컬럼이 없습니다. 분할 기준 시각이 있어야 합니다.")

    ordered = frame.sort_values(time_column, kind="mergesort").reset_index(drop=True)
    n = len(ordered)
    cut = n - int(n * ratio)
    if conf.get("keep_same_time_together", True):
        cut = _advance_past_ties(ordered[time_column], cut)

    head, tail = ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()
    if head.empty or tail.empty:
        raise ValueError(
            f"{n:,}행을 {ratio}로 나누면 한쪽이 빕니다. 비율이나 입력 크기를 확인하세요."
        )
    last, first = head[time_column].max(), tail[time_column].min()
    if last >= first:
        raise AssertionError(f"앞뒤 시각이 겹칩니다: {last} >= {first}. 미래 정보가 과거로 샙니다.")
    return head, tail


def _assert_ordered(split: Split, time_column: str) -> None:
    """미래가 과거로 새지 않았는지 확인한다. 이 검사가 깨지면 실험 결과가 무효다."""
    for earlier, later, name in (
        (split.train, split.val, "학습셋과 검증셋"),
        (split.val, split.test, "검증셋과 평가셋"),
    ):
        if earlier.empty or later.empty:
            raise ValueError(f"{name} 중 한쪽이 비었습니다. 비율 설정을 확인하세요.")
        last, first = earlier[time_column].max(), later[time_column].min()
        if last >= first:
            raise AssertionError(
                f"{name}의 시각이 겹칩니다: {last} >= {first}. 미래 정보가 과거로 샙니다."
            )
