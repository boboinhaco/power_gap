"""
FastAPI 앱 진입점 — 제주 전력수요예측 오차(갭) 사전경보 서빙 & AIOps.

Day1: predict/health 라우터, startup에서 로딩 모드 분기
Day2: data 라우터 (정규화 전력 갭 CSV 업로드)
Day3: "aiops" 로거 → logs/aiops.log, logs 라우터, 이력 재생(/predict/replay) → 드리프트 → 재학습

정적 대시보드(serving_app/static/index.html)는 API 라우터를 먼저 등록한 뒤 "/"에 마지막으로 mount한다.
MVP는 단일 워커 전제다 (예측 이력·드리프트 카운터가 프로세스 메모리에 있음).
"""
import logging
import os
import time

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles

from serving_app import model_loader, reqlog
from serving_app.routers import data, health, logs, metrics, predict

_LOG_DIR = "logs"
os.makedirs(_LOG_DIR, exist_ok=True)
_aiops_logger = logging.getLogger("aiops")
_aiops_logger.setLevel(logging.INFO)
if not _aiops_logger.handlers:
    _handler = logging.FileHandler(os.path.join(_LOG_DIR, "aiops.log"), encoding="utf-8")
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _aiops_logger.addHandler(_handler)
    _aiops_logger.addHandler(logging.StreamHandler())

app = FastAPI(title="제주 전력수요예측 오차 사전경보 · Serving & AIOps")

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(data.router)
app.include_router(logs.router)
app.include_router(metrics.router)  # 대시보드: 운영 지표·레지스트리·알람

_LOGGED_PREFIXES = ("/predict", "/data/upload", "/data/history")  # 운영 지표에 집계하는 요청 (대시보드 폴링은 제외)


@app.middleware("http")
async def request_logger(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    path = request.url.path
    if path.startswith(_LOGGED_PREFIXES):
        reqlog.record(path, response.status_code, (time.perf_counter() - start) * 1000)
    return response

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/", StaticFiles(directory=_STATIC_DIR, html=True), name="static")


@app.on_event("startup")
def startup():
    if os.getenv("LOADING_MODE", "lazy") == "eager":
        model_loader.load_eager()
    else:
        print("[lazy] 모델은 첫 /predict 요청이 들어올 때 로드됩니다.")
