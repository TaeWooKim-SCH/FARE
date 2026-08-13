---
name: 실험
about: 측정을 목적으로 하는 작업
title: '[EXP] '
labels: experiment
---

## 목적

무엇을 알아내려는가. 한 문장.

## 측정 항목

- [ ] Recall / Precision / PR-AUC
- [ ] 변경 항목 수 (공격 실험)
- [ ] 질의 횟수 (공격 실험)
- [ ] 임계값 τ 기록

## 예상 결과와 해석

| 결과 | 해석 |
|---|---|
| | |

예상과 다른 결과도 결과다. 미리 적어두고 시작한다.

## 검증

- [ ] 시간순 분할 확인 (`leakage-auditor`)
- [ ] 도메인 제약 확인 (`constraint-checker`) — 공격 실험인 경우
- [ ] 결과 타당성 확인 (`result-reviewer`)

## 산출물

`results/{실험명}_{날짜}/` — config.yaml, metrics.json, log.txt
