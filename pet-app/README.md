# 반려동물 품종 분류

사진을 업로드하면 ResNet18 모델이 개·고양이 10개 품종 중 상위 3개 후보와 모델 점수를 표시합니다.

## Windows에서 실행

최신 증강 모델은 검증 정확도 94.50%, Macro-F1 0.9444를 기록했습니다. 가중치는 models/best.pt에 배치합니다.
모델 SHA-256은 d30b8eec39ae3fdf7116566d05d0077ced6659486772fd3a3619f714942df377입니다.

PowerShell에서 `app.py`가 있는 폴더를 열고 실행합니다.

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

터미널에 표시된 Local URL을 브라우저에서 열고 JPG 또는 PNG 사진을 업로드합니다.

## Streamlit Community Cloud 배포

GitHub의 dohyeon-app 브랜치에 pet-app 폴더를 업데이트합니다.
models/best.pt도 커밋합니다. 모델은 약 45MB로 브라우저 업로드 한도를 넘으므로 GitHub Desktop 또는 git push를 사용합니다.
Streamlit Community Cloud에서 저장소와 브랜치 dohyeon-app, Main file path pet-app/app.py를 선택합니다.
Advanced settings의 Python은 3.11을 선택한 뒤 Deploy를 실행합니다.
pet-app/requirements.txt가 앱 의존성을 제공하며 모델은 app.py 기준 상대 경로로 읽습니다.

## 모델별 성능 기록

| 모델 | 검증 정확도 | 검증 Macro-F1 | 시험 정확도 | 시험 Macro-F1 |
|---|---:|---:|---:|---:|
| 증강 없는 추가 학습 모델 | 93.00% | 0.9293 | 93.50% | 0.9349 |
| 현재 앱의 flip_color 모델 | 94.50% | 0.9444 | 평가 결과 없음 | 평가 결과 없음 |

현재 모델은 같은 90.75% 기준 체크포인트에서 학습률 0.0001, 배치 32, 시드 42로 추가 학습했습니다.
학습 이미지에 좌우 반전 p=0.5와 밝기·대비 변화 0.1을 적용했고 검증 손실이 가장 낮은 추가 11에포크 모델을 선택했습니다.
검증 400장 중 378장을 맞혔습니다. 이전 모델의 시험 93.50%는 현재 증강 모델의 시험 성능과 구분합니다.

## 평가 재현

팀 데이터 백업 폴더 안에 `raw`와 `dataset/val.csv`, `dataset/test.csv`가 있어야 합니다. 아래 경로를 실제 백업 위치로 바꿉니다. 학습 때 고정한 목록과 이미지 해시를 확인한 후 평가합니다.

```powershell
.\.venv\Scripts\python.exe validate_model.py --backup "C:\path\to\data-backup" --split val --output results/validation

.\.venv\Scripts\python.exe validate_model.py --backup "C:\path\to\data-backup" --split test --output results/final_test
```

각 출력 폴더에 `metrics.json`, `predictions.csv`, `per_class.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, `errors.csv`, `classes.json`이 생성됩니다. 혼동 행렬의 행은 실제 품종, 열은 예측 품종입니다. 기존 결과를 보존하기 위해 재실행 시 새로운 출력 폴더를 지정합니다. 데이터 백업은 팀에서 별도로 제공합니다.
