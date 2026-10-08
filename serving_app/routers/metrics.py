"""
대시보드용 운영 지표·모델 레지스트리·알람 조회 (읽기 전용).

  GET /metrics/summary?window=300   : 요청 수·평균 응답시간·성공률(요청 로그), 버전별 RMSE, 드리프트 점수
  GET /models/versions              : MLflow 레지스트리의 JejuGapPredictor 전 버전 (모드·RMSE·champion 여부)
  GET /models/current               : champion 상세
  GET /alerts/recent?n=20           : aiops.log 최근 알람을 파싱해 반환
  GET /data/raw-files, /data/quality, /data/uploads : Datasets 탭용
"""
import datetime as dt
import os
import re
import time

from fastapi import APIRouter

from serving_app import model_loader, reqlog, state
from serving_app.evaluation import load_thresholds
from serving_app.monitoring.drift_detector import WINDOW_SIZE, evaluate, threshold_mw

router = APIRouter()
_STARTED = time.time()
_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \[(\w+)\] (\[(\w+)\] )?(.*)$")


def _registry_versions() -> list[dict]:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        c = MlflowClient()
        champ = model_loader.champion_version()
        out = []
        for v in c.search_model_versions(f"name='{model_loader.MODEL_NAME}'"):
            run = c.get_run(v.run_id) if v.run_id else None
            params = run.data.params if run else {}
            metrics = run.data.metrics if run else {}
            out.append({
                "version": int(v.version), "created": v.creation_timestamp / 1000,
                "mode": params.get("mode", "-"), "rmse": metrics.get("rmse"),
                "champion_rmse": metrics.get("champion_rmse"), "gate": v.tags.get("gate", ""),
                "train": params.get("train", ""), "val": params.get("val", ""),
                "is_champion": str(v.version) == champ,
                "val_metrics": {k: metrics[k] for k in ("val_rmse_zero", "val_rmse_train_mean", "val_rmse_persistence",
                                                        "val_rmse_lstm", "val_dirhit_lstm", "gate_improvement_pct",
                                                        "val_mae_lstm", "val_wape_lstm", "val_bias_lstm") if k in metrics},
            })
        return sorted(out, key=lambda x: -x["version"])
    except Exception as e:  # 레지스트리가 비어 있거나 local 모드
        return []


@router.get("/metrics/summary")
def metrics_summary(window: int = 300):
    s = reqlog.summary(max(60, min(window, 86400)))
    versions = _registry_versions()
    basis = state.last_replay_end  # 가장 최근 재생 구간의 마지막 날 기준으로 평가 (여러 구간을 재생해도 최신 재생을 따른다)
    records = state.confirmed_records(upto=basis) if basis else state.confirmed_records()
    drift = evaluate(records) if records else None
    th = threshold_mw()
    return {
        **s,
        "rmse_by_version": [{"version": v["version"], "rmse": v["rmse"]} for v in reversed(versions) if v["rmse"] is not None],
        "drift": None if not drift else {
            "rolling_rmse_mw": drift.get("rolling_rmse_mw"), "threshold_mw": th,
            "rolling_mae_mw": drift.get("rolling_mae_mw"), "rolling_wape": drift.get("rolling_wape"),
            "mae_threshold_mw": drift.get("mae_threshold_mw"), "rule": drift.get("rule"),
            "breach_rmse": drift.get("breach_rmse"), "breach_mae": drift.get("breach_mae"),
            "rolling_bias_mw": drift.get("rolling_bias_mw"),
            "ratio": (drift.get("rolling_rmse_mw") or 0) / th if th else None,
            "breach": drift.get("breach"), "ready": drift.get("ready"), "n_records": drift.get("n_records"),
            "breach_streak_days": state.breach_streak_days, "breach_streak_start": state.breach_streak_start,
            "as_of": drift.get("date"), "window": WINDOW_SIZE, "basis": basis,
        },
        "last_retrain": state.last_retrain,
    }


@router.get("/metrics/validation")
def metrics_validation():
    """저장된 walk-forward 평가(docs/eval_*.json)로 모델별 RMSE·MAE·WAPE·Bias·방향 적중률을 돌려준다.
    WAPE = Σ|실제 갭 − 예측 갭| / Σ|실제 갭| → naive(zero)는 정의상 1.0이므로 'KPX 오차 중 남은 비율'로 읽는다."""
    import glob
    import json
    from data.metrics import summarize
    out = []
    for p in sorted(glob.glob("docs/eval_*.json")):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        actual = d.get("actual") or []
        preds = d.get("predictions") or {}
        if not actual or not preds:
            continue
        out.append({"start": d.get("start"), "end": d.get("end"), "n": len(actual), "file": os.path.basename(p),
                    "metrics": {m: summarize(actual, v) for m, v in preds.items() if len(v) == len(actual)}})
    return {"periods": out}


@router.get("/models/versions")
def model_versions():
    return {"name": model_loader.MODEL_NAME, "versions": _registry_versions()}


@router.get("/models/current")
def model_current():
    versions = _registry_versions()
    champ = next((v for v in versions if v["is_champion"]), None)
    th = load_thresholds() or {}
    return {"name": model_loader.MODEL_NAME, "champion": champ, "loaded_version": model_loader.current_version(),
            "loaded": model_loader.is_loaded(), "source": os.getenv("MODEL_SOURCE", "local"),
            "thresholds": th, "uptime_seconds": time.time() - _STARTED}


@router.get("/alerts/recent")
def logs_recent(n: int = 20):
    path = "logs/aiops.log"
    if not os.path.exists(path):
        return {"alerts": []}
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    out = []
    for line in reversed(lines):
        m = _LINE.match(line.rstrip("\n"))
        if not m:
            continue
        ts = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").timestamp()
        out.append({"ts": ts, "level": m.group(4) or m.group(2), "message": m.group(5)})
    return {"alerts": out}


@router.get("/data/raw-files")
def raw_files():
    raw = os.getenv("RAW_DIR", "data/raw")
    if not os.path.isdir(raw):
        return {"files": []}
    import unicodedata
    files = []
    for f in sorted(os.listdir(raw)):
        if f.startswith("."):
            continue
        p = os.path.join(raw, f)
        nf = unicodedata.normalize("NFC", f)
        kind = ("제주 예측" if nf.lower().startswith("pssjeju") else "제주 실적" if "제주전력수요" in nf
                else "기타(미사용)")
        files.append({"name": nf, "size": os.path.getsize(p), "kind": kind, "mtime": os.path.getmtime(p)})
    return {"files": files}


@router.get("/data/quality")
def quality():
    p = "data/quality_report.md"
    return {"report": open(p, encoding="utf-8").read() if os.path.exists(p) else "(아직 생성되지 않음)"}


@router.get("/data/uploads")
def uploads():
    from data.storage import UPLOAD_DIR, UPLOAD_PREFIX
    if not os.path.isdir(UPLOAD_DIR):
        return {"files": []}
    fs = [f for f in os.listdir(UPLOAD_DIR) if f.startswith(UPLOAD_PREFIX) and f.endswith(".csv")]
    out = [{"name": f, "size": os.path.getsize(os.path.join(UPLOAD_DIR, f)), "mtime": os.path.getmtime(os.path.join(UPLOAD_DIR, f))} for f in fs]
    return {"files": sorted(out, key=lambda x: -x["mtime"])}
