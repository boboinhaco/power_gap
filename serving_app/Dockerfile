# 제주 전력수요예측 오차 사전경보 서빙 앱 (단일 컨테이너, 포트 8099)
# 빌드/실행은 프로젝트 루트에서: docker compose -f serving_app/docker-compose.yml up --build
#
# 이미지 빌드 시점에 컨테이너 안에서 전부 끝낸다:
#   data/raw 원본 → prepare_power_gap.py(정규화 CSV + 품질 보고서, uploads 시드)
#   → train_baseline_v1.py(2024 스케일러·로컬 모델·thresholds.json)
#   → train_and_register.py(MLflow sqlite 기록, 검증 walk-forward 게이트, champion alias)
# GATE_OVERRIDE=1: 게이트(naive 최저 RMSE)에 떨어져도 시연용으로 승격 (MLflow 태그 gate=override 로 남음). 기본 0.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY data/ data/
COPY scripts/ scripts/
COPY serving_app/ serving_app/
COPY docs/eval_*.json docs/

ARG GATE_OVERRIDE=0
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db
RUN python data/prepare_power_gap.py --seed-upload \
 && python scripts/train_baseline_v1.py \
 && GATE_OVERRIDE=${GATE_OVERRIDE} python serving_app/train_and_register.py

EXPOSE 8099
ENV MODEL_SOURCE=mlflow
ENV LOADING_MODE=eager

# 단일 워커 전제 (예측 이력·드리프트 카운터가 프로세스 메모리에 있음)
CMD ["uvicorn", "serving_app.main:app", "--host", "0.0.0.0", "--port", "8099", "--workers", "1"]
