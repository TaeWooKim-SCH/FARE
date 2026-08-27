"""컬럼 프로파일.

column-reference.md에 적어둔 컬럼별 수치를 코드로 다시 만든다. 문서에 적힌 수치는
일회성 스크립트로 뽑은 값이라 재현되지 않았고, 그 수치가 constraints.yaml과 공격
코드로 흘러가므로 틀리면 아래로 전부 번진다.

전체 기준과 학습셋 기준을 모두 낼 수 있게 만들었다. 데이터를 기술하는 것(문서)은
전체 기준이어도 되지만, 무엇을 버릴지 같은 **결정**은 학습셋만 보고 내려야 한다.
전체를 보고 정하면 그 판단 자체가 미래를 본 것이 된다.
"""

from __future__ import annotations

import hashlib
import re

import pandas as pd
from pandas.api.types import is_numeric_dtype

# 식별자와 라벨은 feature가 아니다. TransactionID는 순서가 TransactionDT와 묶여 있어
# 모델 입력에서도 뺀다(feature-taxonomy.md 3절).
DEFAULT_EXCLUDE = ("TransactionID", "isFraud")

_V_PATTERN = re.compile(r"^V(\d+)$")


def column_profile(df: pd.DataFrame, exclude: tuple[str, ...] = DEFAULT_EXCLUDE) -> pd.DataFrame:
    """컬럼마다 dtype·결측률·고유값 수·최빈값 비율과 수치 요약을 낸다.

    `top_share`는 결측을 하나의 값으로 쳤을 때 최빈값이 차지하는 비율이다. 결측을
    빼고 세면 결측 99%짜리 컬럼이 "값이 고르게 퍼져 있다"로 보여서 오해를 부른다.
    """
    n = len(df)
    if n == 0:
        raise ValueError("빈 데이터프레임은 프로파일을 낼 수 없습니다.")

    rows = []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        nonnull = s.dropna()
        counts = s.value_counts(dropna=False)
        row = {
            "column": col,
            "dtype": str(s.dtype),
            "missing": float(s.isna().mean()),
            "nunique": int(nonnull.nunique()),
            "top_value": counts.index[0] if len(counts) else None,
            "top_share": float(counts.iloc[0] / n) if len(counts) else 0.0,
            "is_numeric": bool(is_numeric_dtype(s)),
        }
        if row["is_numeric"] and len(nonnull):
            q = nonnull.quantile([0.25, 0.5, 0.75])
            row.update(
                min=float(nonnull.min()),
                q25=float(q.iloc[0]),
                median=float(q.iloc[1]),
                q75=float(q.iloc[2]),
                max=float(nonnull.max()),
                # 금액 제약에 직접 걸린다. 정수만인 컬럼에 소수를 넣으면 바로 걸린다.
                integer_only=bool((nonnull == nonnull.round()).all()),
                has_negative=bool((nonnull < 0).any()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def constant_columns(profile: pd.DataFrame) -> list[str]:
    """값이 한 종류뿐인 컬럼.

    트리 모델은 이 컬럼으로 분기를 만들 수 없어 버리든 두든 출력이 같고, 스케일러는
    표준편차가 0이라 나눗셈에서 터진다. 어떤 컬럼이 여기 해당하는지는 학습셋 기준으로
    판정해야 한다 — 전체를 보고 정하면 그 판단 자체가 미래를 본 것이 된다.
    """
    if "nunique" not in profile.columns or "column" not in profile.columns:
        raise KeyError("column_profile()이 낸 결과를 넣어야 합니다(column·nunique 필요).")
    return sorted(profile.loc[profile["nunique"] <= 1, "column"].tolist())


def _mask_key(mask: pd.Series) -> str:
    """결측 마스크를 해시로 줄인다. 결측 '개수'가 같아도 위치가 다르면 다른 블록이다."""
    return hashlib.md5(mask.to_numpy().tobytes()).hexdigest()


def missing_pattern_blocks(df: pd.DataFrame, prefix: str = "V") -> pd.DataFrame:
    """결측 패턴이 완전히 같은 컬럼끼리 묶는다.

    V 339개를 개별로 판정할 수는 없으나, 결측 패턴이 같은 묶음은 같은 엔티티에서
    파생된 한 덩어리로 볼 수 있다. 이 묶음이 constraints.yaml의 기술 단위가 된다.
    """
    cols = [c for c in df.columns if _V_PATTERN.match(c)] if prefix == "V" else [
        c for c in df.columns if c.startswith(prefix)
    ]
    if not cols:
        raise ValueError(f"{prefix}로 시작하는 컬럼이 없습니다.")

    n = len(df)
    groups: dict[str, list[str]] = {}
    for col in cols:
        groups.setdefault(_mask_key(df[col].isna()), []).append(col)

    rows = []
    for members in groups.values():
        numbers = sorted(int(_V_PATTERN.match(c).group(1)) for c in members) if prefix == "V" else []
        # 번호가 끊기지 않고 이어지는지. 끊기면 constraints.yaml에 구간으로 적을 수 없다.
        contiguous = bool(numbers) and (numbers[-1] - numbers[0] + 1 == len(numbers))
        rows.append(
            {
                "n_columns": len(members),
                "missing_count": int(df[members[0]].isna().sum()),
                "missing_rate": float(df[members[0]].isna().mean()),
                "contiguous": contiguous,
                "columns": ",".join(sorted(members, key=lambda c: int(_V_PATTERN.match(c).group(1))))
                if prefix == "V"
                else ",".join(sorted(members)),
            }
        )
    out = pd.DataFrame(rows).sort_values(["missing_count", "n_columns"]).reset_index(drop=True)
    if n and out["missing_count"].max() > n:
        raise ValueError("결측 수가 행 수를 넘습니다. 입력을 확인하세요.")
    return out


def productcd_gating(
    df: pd.DataFrame, group_column: str = "ProductCD", exclude: tuple[str, ...] = DEFAULT_EXCLUDE
) -> pd.DataFrame:
    """상품 코드별로 통째로 비어 있는 컬럼을 찾는다.

    ProductCD가 feature의 존재 자체를 가른다. 이 비대칭 때문에 D9 기반 정합성 검사가
    W에서는 발동조차 하지 않으므로, 실험 결과를 상품별로 쪼개 보고해야 한다
    (feature-taxonomy.md 5절).
    """
    if group_column not in df.columns:
        raise KeyError(f"{group_column} 컬럼이 없습니다.")

    targets = [c for c in df.columns if c not in exclude and c != group_column]
    rows = []
    for value, part in df.groupby(group_column, dropna=False):
        all_missing = [c for c in targets if part[c].isna().all()]
        rows.append(
            {
                "product": value,
                "rows": len(part),
                "n_all_missing": len(all_missing),
                "columns": ",".join(all_missing),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)
