# FARE: Fraud Adversarial Robustness Evaluation

금융 사기 탐지 모델의 적대적 강건성 평가 및 방어 기법 적용

> 사기꾼이 현실적으로 조작 가능한 항목만 바꿔서 사기 탐지 모델을 회피할 수 있는가?
> 가능하다면 표준적 방어로 얼마나 복구되며, 그 대가는 무엇인가?

순천향대학교 AI·빅데이터학과 학부 연구 · 진행 중

---

## 배경

사기 탐지 모델은 보통 테스트셋 정확도로 평가한 뒤 배포된다. 여기에는 검증되지 않은 가정이 있다 — **사기꾼이 모델의 존재와 무관하게 행동한다는 것.**

실제로는 그렇지 않다. 탐지 모델이 있다는 걸 아는 사기꾼은 큰 금액을 여러 건으로 쪼개고 시간대를 평범한 때로 옮긴다. 즉 **모델의 존재가 입력 데이터를 바꾼다.** 따라서 테스트셋 성능은 평균적 성능일 뿐, 적응하는 공격자 앞에서의 최악 성능을 보장하지 않는다.

이미지와 달리 거래는 자유롭게 변형할 수 없다. 공격자는 금액과 시각을 정할 수 있지만, 카드 정보나 과거 이력 기반 집계값은 바꿀 수 없다. **이 제약을 강제하는 것이 이 연구의 핵심이다.**

---

## 구성

| 단계 | 내용 |
|---|---|
| 1. 탐지 | XGBoost, Random Forest, MLP — 시간순 분할로 학습 |
| 2. 공격 | 규칙 기반 회피, Decision-Based 공격 (도메인 제약 하) |
| 3. 전이성 | 한 모델을 공격해 만든 거래를 다른 모델에 투입 |
| 4. 방어 | 적대적 학습 — 강건성 회복과 평상시 성능 손실 측정 |

---

## 평가 원칙

결과의 신뢰성을 좌우하는 두 가지를 코드로 강제한다.

**시간순 분할** — `TransactionDT` 기준으로 나눈다. 무작위 분할은 미래 정보가 과거로 새어 모든 지표를 부풀린다.

**운영 임계값** — 0.5가 아니다. 사기율이 3.5%라 모델이 대부분 낮은 점수를 주므로, 실제 시스템은 훨씬 낮은 τ를 쓴다. 공격 라이브러리는 0.5를 가정해 조기 종료하므로, 검증셋에서 정한 τ를 공격에 주입해야 한다.

**지표** — 정확도는 쓰지 않는다(전부 정상이라 찍어도 96.5%). Recall을 주 지표로 하고, Precision·PR-AUC와 함께 **변경한 항목 수**(공격 비용)를 보고한다.

---

## 위협 모델

| | |
|---|---|
| 공격 유형 | 회피(Evasion) — 학습된 모델 대상, 학습 데이터 접근 불가 |
| 공격자 지식 | Black box — 승인/거절 결과만 관찰 |
| 조작 가능 | 거래 금액, 거래 시각 |
| 조작 불가 | 카드 식별 정보, 주소, 과거 이력 기반 집계값 |
| 추가 제약 | 변형 후에도 사기로서 유효해야 함 |

IEEE-CIS 각 항목의 조작 가능 여부는 `config/constraints.yaml`에 정의하고, 생성된 모든 거래를 이 기준으로 검증한다.

---

## 데이터

[IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) — 590,540건, 사기율 3.5%, 393개 feature.

저장소에 포함하지 않는다. Kaggle에서 받아 `datasets/`에 둔다.

이후 BAF(은행 계좌), Elliptic(암호화폐 그래프)으로 확장 예정. 데이터 로딩 계층을 분리해 나머지 파이프라인을 재사용할 수 있게 작성한다.

---

## 구조

```
.venv/          가상환경
config/         제약 정의, 모델·공격 설정
datasets/       원본 데이터 (git 제외)
docs/           연구 계획서, feature 분류 근거
src/
  data/         로딩, 시간순 분할
  models/       탐지 모델 학습, 임계값 결정
  attacks/      규칙 기반, Decision-Based, 제약 강제
  defense/      적대적 학습
  eval/         지표, 전이성
scripts/        실험 실행 진입점
notebooks/      탐색적 분석, 결과 시각화
results/        실험 결과 (git 제외)
```

---

## 진행 상황

- [ ] IEEE-CIS 데이터 확보 및 feature 3분류
- [ ] 시간순 분할 및 탐지 모델 3종 학습
- [ ] 운영 임계값 결정
- [ ] 규칙 기반 공격
- [ ] Decision-Based 공격
- [ ] 전이성 실험
- [ ] 적대적 학습 방어
- [ ] 결과 정리 및 논문 작성

---

## 참고 문헌

- Carminati et al., *Evasion Attacks against Banking Fraud Detection Systems*, RAID 2020
- Cartella et al., *Adversarial Attacks for Tabular Data*, 2021
- Lunghi et al., *Adversarial Learning in Real-World Fraud Detection*, DEC 2023
- Lunghi et al., *Fraud-RLA*, IEEE TDSC 2026

## License

MIT