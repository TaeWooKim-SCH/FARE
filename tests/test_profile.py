"""컬럼 프로파일 검증.

이 프로파일이 taxonomy 부록의 수치를 다시 만드는 근거이므로, 세는 방식이 바뀌면
문서와 어긋난다. 세는 규칙을 테스트로 고정한다.
"""

import numpy as np
import pandas as pd
import pytest

from src.eda.profile import column_profile, missing_pattern_blocks, productcd_gating


def test_값이_한_종류뿐인_컬럼은_고유값이_1로_잡힌다():
    prof = column_profile(pd.DataFrame({"a": [1.0, 1.0, 1.0, 1.0]}))
    assert prof.loc[0, "nunique"] == 1
    assert prof.loc[0, "top_share"] == 1.0


def test_결측을_하나의_값으로_쳐서_최빈값_비율을_센다():
    # 결측을 빼고 세면 "값이 고르게 퍼졌다"로 보여서 결측 위주 컬럼을 놓친다.
    frame = pd.DataFrame({"a": [1.0, 2.0] + [np.nan] * 8})
    prof = column_profile(frame)
    assert prof.loc[0, "missing"] == pytest.approx(0.8)
    assert prof.loc[0, "top_share"] == pytest.approx(0.8)
    assert prof.loc[0, "nunique"] == 2


def test_수치형은_사분위와_정수_여부와_음수_여부를_낸다():
    prof = column_profile(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))
    assert prof.loc[0, "is_numeric"]
    assert prof.loc[0, "median"] == pytest.approx(2.5)
    assert prof.loc[0, "integer_only"]
    assert not prof.loc[0, "has_negative"]


def test_소수가_섞이면_정수_전용이_아니라고_표시된다():
    # 금액 제약이 여기 걸린다. H·R은 비정수가 한 건도 없어야 한다.
    prof = column_profile(pd.DataFrame({"a": [1.0, 2.5, 3.0]}))
    assert not prof.loc[0, "integer_only"]


def test_문자열_컬럼은_사분위를_내지_않는다():
    prof = column_profile(pd.DataFrame({"a": ["W", "C", "W"]}))
    assert not prof.loc[0, "is_numeric"]
    assert pd.isna(prof.loc[0, "median"]) if "median" in prof.columns else True
    assert prof.loc[0, "top_value"] == "W"


def test_식별자와_라벨은_프로파일_대상이_아니다():
    frame = pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1], "a": [1.0, 2.0]})
    prof = column_profile(frame)
    assert prof["column"].tolist() == ["a"]


def test_빈_데이터프레임은_거부한다():
    with pytest.raises(ValueError, match="빈 데이터프레임"):
        column_profile(pd.DataFrame({"a": []}))


def test_결측_개수가_같아도_위치가_다르면_다른_블록이다():
    # 결측 '개수'로 묶으면 이 둘이 한 블록으로 잘못 합쳐진다.
    frame = pd.DataFrame(
        {
            "V1": [1.0, np.nan, 3.0, 4.0],
            "V2": [1.0, 2.0, np.nan, 4.0],
        }
    )
    blocks = missing_pattern_blocks(frame)
    assert len(blocks) == 2


def test_결측_위치가_같으면_한_블록으로_묶인다():
    frame = pd.DataFrame(
        {
            "V1": [1.0, np.nan, 3.0],
            "V2": [9.0, np.nan, 7.0],
        }
    )
    blocks = missing_pattern_blocks(frame)
    assert len(blocks) == 1
    assert blocks.loc[0, "n_columns"] == 2
    assert blocks.loc[0, "columns"] == "V1,V2"


def test_번호가_이어지면_연속이라고_표시한다():
    frame = pd.DataFrame({f"V{i}": [1.0, np.nan] for i in (1, 2, 3)})
    blocks = missing_pattern_blocks(frame)
    assert bool(blocks.loc[0, "contiguous"])


def test_번호가_끊기면_연속이_아니라고_표시한다():
    # 끊긴 블록은 constraints.yaml에 번호 구간으로 적을 수 없다.
    frame = pd.DataFrame({f"V{i}": [1.0, np.nan] for i in (1, 2, 5)})
    blocks = missing_pattern_blocks(frame)
    assert not bool(blocks.loc[0, "contiguous"])


def test_V로_시작하는_컬럼이_없으면_거부한다():
    with pytest.raises(ValueError, match="컬럼이 없습니다"):
        missing_pattern_blocks(pd.DataFrame({"a": [1.0]}))


def test_상품별로_통째로_비어_있는_컬럼을_찾는다():
    frame = pd.DataFrame(
        {
            "ProductCD": ["W", "W", "C", "C"],
            "only_in_w": [1.0, 2.0, np.nan, np.nan],
            "always": [1.0, 2.0, 3.0, 4.0],
        }
    )
    gating = productcd_gating(frame)
    by_product = dict(zip(gating["product"], gating["columns"]))
    assert by_product["C"] == "only_in_w"
    assert by_product["W"] == ""


def test_상품_컬럼이_없으면_거부한다():
    with pytest.raises(KeyError):
        productcd_gating(pd.DataFrame({"a": [1.0]}))
