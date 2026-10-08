"""Day1: 헬스체크 — 모델 로딩 상태·버전, 마지막 실적 수신일, 드리프트 임계값, 데이터 파일."""
import os

from fastapi import APIRouter

from data.features import SEQ_LEN
from serving_app import model_loader, state
from serving_app.monitoring.drift_detector import threshold_mw

router = APIRouter()


@router.get("/health")
def health():
    source = os.getenv("MODEL_SOURCE", "local")
    version = model_loader.current_version()
    if version is None and source == "mlflow":
        v = model_loader.champion_version()
        version = f"{model_loader.MODEL_NAME} v{v} (not loaded yet)" if v else None
    try:
        from data.storage import latest_upload
        data_file = os.path.basename(latest_upload())
    except FileNotFoundError:
        data_file = None
    return {
        "status": "ok",
        "model_loaded": model_loader.is_loaded(),
        "model_version": version,
        "model_source": source,
        "loading_mode": os.getenv("LOADING_MODE", "lazy"),
        "last_actual_date": state.last_actual_date,
        "last_replay_end": state.last_replay_end,
        "drift_rmse_threshold_mw": round(threshold_mw(), 1),
        "data_file": data_file,
        "seq_len": SEQ_LEN,
        "breach_streak_days": state.breach_streak_days,
    }
