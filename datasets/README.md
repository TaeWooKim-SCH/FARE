# datasets/

용량 문제로 데이터 파일은 저장소에 포함하지 않습니다. 아래에서 받아 각 폴더에 배치하세요.

## ieee-cis/

**IEEE-CIS Fraud Detection** — 주력 데이터
https://www.kaggle.com/c/ieee-fraud-detection

590,540건 · 사기율 3.5% · 393 feature

```
ieee-cis/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

대회 test set은 라벨이 없으므로, `train_transaction.csv`를 `TransactionDT` 기준으로 시간순 분할하여 학습·평가에 사용합니다.

## baf/

**Bank Account Fraud (NeurIPS 2022)** — 일반화 검증용
https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

100만 건 × 6종 변형 · feature 의미 공개 · 시간 축(`month`) 포함