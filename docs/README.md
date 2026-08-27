# 문서 안내

어떤 문서를 언제 보는지 적어둔다.

| 문서 | 무엇이 들어 있나 | 언제 보나 |
|---|---|---|
| [research-plan.md](research-plan.md) | 이 연구를 왜 하고 어떻게 할 건지. 평가 방법, 예상 반론, 위험 요소 | 작업을 시작하기 전에. 새 단계에 들어갈 때 |
| [feature-taxonomy.md](feature-taxonomy.md) | 컬럼 400여 개 중 공격자가 무엇을 바꿀 수 있고 무엇을 못 바꾸는지, 그리고 그 근거 | 공격 대상을 정할 때. 이 연구의 기준이 되는 문서 |
| [attack-constraints.md](attack-constraints.md) | 금액과 시각을 바꿀 때 지켜야 할 조건과, 바꾸면 같이 고쳐야 하는 컬럼 | 공격 코드를 짤 때. `config/constraints.yaml`이 여기서 나온다 |
| [column-reference.md](column-reference.md) | 컬럼 하나하나를 직접 집계한 기록과 출처 | "이 컬럼이 뭐였지" 할 때 찾아본다 |
| [defense-notes.md](defense-notes.md) | 방어(적대적 학습)에서 미리 걸릴 문제와 만들면 안 되는 검사 | 방어 단계에 들어갈 때 |

## 처음 보는 사람이라면

`research-plan.md`로 무엇을 하려는 연구인지 보고, `feature-taxonomy.md`의 3절 분류표를 보면 데이터의 큰 그림이 잡힌다. 나머지 셋은 필요할 때 찾아보면 된다.

## 문서를 고칠 때

수치를 적을 때는 어디서 나온 값인지 함께 적는다. `column-reference.md`의 값은 `scripts/eda/run_profile.py`로 다시 만들 수 있고, 핵심 주장은 `tests/test_taxonomy_claims.py`가 검사한다. **문서와 데이터가 어긋나면 데이터가 맞다고 보고 문서를 고친다.**
