"""시간순 분할 검증.

분할이 깨지면 이 연구의 모든 수치가 무효가 되므로, 규칙을 코드로 고정한다.
"""

import pandas as pd
import pytest

from src.data.split import split_tail, time_split

BASE_CONFIG = {
    "split": {
        "time_column": "TransactionDT",
        "train_ratio": 0.7,
        "val_ratio": 0.1,
        "keep_same_time_together": True,
    }
}


def make_config(**overrides) -> dict:
    cfg = {"split": dict(BASE_CONFIG["split"])}
    cfg["split"].update(overrides)
    return cfg


def make_frame(times) -> pd.DataFrame:
    return pd.DataFrame({"TransactionDT": list(times), "isFraud": 0})


def test_학습셋의_마지막_거래는_검증셋의_첫_거래보다_앞선다():
    split = time_split(make_frame(range(100)), make_config())
    assert split.train["TransactionDT"].max() < split.val["TransactionDT"].min()


def test_검증셋의_마지막_거래는_평가셋의_첫_거래보다_앞선다():
    split = time_split(make_frame(range(100)), make_config())
    assert split.val["TransactionDT"].max() < split.test["TransactionDT"].min()


def test_분할해도_전체_행_수가_보존된다():
    split = time_split(make_frame(range(100)), make_config())
    assert len(split) == 100


def test_설정한_비율대로_잘린다():
    split = time_split(make_frame(range(1000)), make_config())
    assert len(split.train) == 700
    assert len(split.val) == 100
    assert len(split.test) == 200


def test_입력이_시간순이_아니어도_정렬한_뒤_자른다():
    shuffled = make_frame([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])
    split = time_split(shuffled, make_config(train_ratio=0.6, val_ratio=0.2))
    assert list(split.train["TransactionDT"]) == [0, 1, 2, 3, 4, 5]
    assert list(split.test["TransactionDT"]) == [8, 9]


def test_같은_시각_거래는_경계를_사이에_두고_갈리지_않는다():
    # 학습셋 경계(index 70)가 시각 68을 공유하는 네 건의 한가운데 떨어지도록 만든 입력
    times = list(range(68)) + [68, 68, 68, 68] + list(range(72, 100))
    split = time_split(make_frame(times), make_config())
    assert 68 not in set(split.val["TransactionDT"]) | set(split.test["TransactionDT"])
    assert (split.train["TransactionDT"] == 68).sum() == 4
    assert split.train["TransactionDT"].max() < split.val["TransactionDT"].min()


def test_같은_시각을_묶지_않으면_경계에서_시각이_겹쳐_분할이_거부된다():
    times = [10, 20, 30, 40, 50, 100, 100, 100, 100, 200]
    with pytest.raises(AssertionError, match="학습셋과 검증셋"):
        time_split(make_frame(times), make_config(keep_same_time_together=False))


# 검증셋과 평가셋 사이 경계도 학습셋 경계와 똑같이 막아야 한다.
# 두 경계를 한 테스트로 덮으면 앞쪽에서 먼저 걸려 뒤쪽 방어가 실행되지 않는다.
TIES_AT_VAL_BOUNDARY = list(range(78)) + [78] * 8 + list(range(86, 100))


def test_검증셋_경계의_같은_시각_거래도_갈리지_않는다():
    split = time_split(make_frame(TIES_AT_VAL_BOUNDARY), make_config())
    assert (split.val["TransactionDT"] == 78).sum() == 8
    assert 78 not in set(split.test["TransactionDT"])
    assert split.val["TransactionDT"].max() < split.test["TransactionDT"].min()


def test_검증셋과_평가셋의_시각이_겹치면_분할이_거부된다():
    with pytest.raises(AssertionError, match="검증셋과 평가셋"):
        time_split(
            make_frame(TIES_AT_VAL_BOUNDARY),
            make_config(keep_same_time_together=False),
        )


def test_설정에_키가_없으면_같은_시각을_묶는_쪽이_기본이다():
    # 기본값이 조용히 뒤집히면 누수가 열리므로 기본 동작을 테스트로 고정한다.
    cfg = {"split": {"time_column": "TransactionDT", "train_ratio": 0.7, "val_ratio": 0.1}}
    split = time_split(make_frame(TIES_AT_VAL_BOUNDARY), cfg)
    assert (split.val["TransactionDT"] == 78).sum() == 8


def test_시각이_숫자가_아니면_분할을_거부한다():
    # 문자열이면 사전순 정렬이라 '1000'이 '2'보다 앞서고, 겹침 검사도 그냥 통과한다.
    frame = make_frame(["1", "10", "100", "1000", "2", "20", "200", "3", "30", "300"])
    with pytest.raises(TypeError, match="숫자가 아닙니다"):
        time_split(frame, make_config())


def test_검증셋_비율이_0이면_분할을_거부한다():
    with pytest.raises(ValueError, match="검증셋"):
        time_split(make_frame(range(100)), make_config(val_ratio=0))


def test_시각에_결측이_있으면_분할을_거부한다():
    frame = make_frame([1, 2, None, 4, 5, 6, 7, 8, 9, 10])
    with pytest.raises(ValueError, match="결측"):
        time_split(frame, make_config())


def test_기준_시각_컬럼이_없으면_분할을_거부한다():
    with pytest.raises(KeyError):
        time_split(pd.DataFrame({"isFraud": [0, 1]}), make_config())


def test_평가셋이_남지_않는_비율은_거부한다():
    with pytest.raises(ValueError, match="비율"):
        time_split(make_frame(range(100)), make_config(train_ratio=0.9, val_ratio=0.2))


# ── 학습셋에서 조기 종료용 조각 떼기 ───────────────────────────────────────


def test_뒤쪽을_떼어낸다():
    head, tail = split_tail(make_frame(range(100)), 0.2, make_config())
    assert len(head) == 80
    assert len(tail) == 20
    assert head["TransactionDT"].max() < tail["TransactionDT"].min()


def test_뗀_조각도_시간순으로_뒤쪽이다():
    """무작위로 뽑으면 미래가 학습 쪽으로 샌다."""
    head, tail = split_tail(make_frame(range(100)), 0.3, make_config())
    assert set(tail["TransactionDT"]) == set(range(70, 100))


def test_입력이_섞여_있어도_시간순으로_가른다():
    frame = make_frame([5, 1, 9, 3, 7, 2, 8, 4, 6, 0])
    head, tail = split_tail(frame, 0.3, make_config())
    assert head["TransactionDT"].max() < tail["TransactionDT"].min()
    assert sorted(tail["TransactionDT"]) == [7, 8, 9]


def test_같은_시각이_양쪽으로_갈리지_않는다():
    """경계가 같은 초 한가운데 떨어지면 그 시각이 끝나는 곳까지 민다.

    비율 0.35면 경계가 index 7, 즉 시각 5인 네 건의 한가운데 떨어진다.
    밀지 않으면 앞쪽 마지막 시각과 뒤쪽 첫 시각이 둘 다 5가 된다.
    """
    frame = make_frame([0, 1, 2, 3, 4, 5, 5, 5, 5, 9])
    head, tail = split_tail(frame, 0.35, make_config())
    assert head["TransactionDT"].max() < tail["TransactionDT"].min()
    assert (head["TransactionDT"] == 5).sum() == 4
    assert list(tail["TransactionDT"]) == [9]


def test_경계가_동점_밖에_떨어지면_그대로_둔다():
    """밀 필요가 없는데 미는 버그를 잡는다."""
    frame = make_frame([0, 1, 2, 3, 4, 5, 5, 5, 5, 9])
    head, tail = split_tail(frame, 0.5, make_config())
    assert list(head["TransactionDT"]) == [0, 1, 2, 3, 4]
    assert list(tail["TransactionDT"]) == [5, 5, 5, 5, 9]


def test_비율이_범위를_벗어나면_거부한다():
    for bad in (0, 1, -0.1, 1.5):
        with pytest.raises(ValueError, match="비율이 잘못됐습니다"):
            split_tail(make_frame(range(100)), bad, make_config())


def test_한쪽이_비면_거부한다():
    """조각이 비면 조기 종료가 아예 안 걸리거나 학습할 것이 없어진다."""
    with pytest.raises(ValueError, match="한쪽이 빕니다"):
        split_tail(make_frame(range(3)), 0.01, make_config())


def test_시각_컬럼이_없으면_거부한다():
    with pytest.raises(KeyError, match="TransactionDT"):
        split_tail(pd.DataFrame({"isFraud": [0, 1]}), 0.2, make_config())


def test_원본을_건드리지_않는다():
    frame = make_frame([3, 1, 2])
    split_tail(frame, 0.34, make_config())
    assert list(frame["TransactionDT"]) == [3, 1, 2]
