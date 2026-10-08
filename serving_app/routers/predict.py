"""
예측 API — serving_app/routers/predict.py (명세서 v2 5장)

  POST /predict          : 연속 N일 (gap_mw, forecast_mw) + target_date → 다음 날 갭 1개, 방향, 모델 버전
  POST /predict/replay   : 과거 이력을 날짜순 재생 → 예측·실적 연결 → 롤링 RMSE → (연속 2회 초과 시) 재학습
  GET  /predict/history  : 최근 예측·실적 기록
"""
import datetime as dt
import json
import os
import time

from fastapi import APIRouter

from data.features import SEQ_LEN
from serving_app import model_loader, reqlog, state
from serving_app.evaluation import load_thresholds
from serving_app.monitoring.retrain_trigger import check_and_trigger
from serving_app.schemas import PredictRequest, PredictResponse, ReplayRequest, ReplayResponse

router = APIRouter()
EXCLUDED_PATH = os.getenv("EXCLUDED_DATES_PATH", "data/excluded_dates.json")


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _excluded_in(start: dt.date, end: dt.date) -> list[str]:
    if not os.path.exists(EXCLUDED_PATH):
        return []
    with open(EXCLUDED_PATH, encoding="utf-8") as f:
        ex = json.load(f)
    return sorted(d for d in ex if start.isoformat() <= d <= end.isoformat())


def direction_of(pred: float) -> str:
    th = load_thresholds() or {}
    under, over = th.get("direction_under_mw", -30.0), th.get("direction_over_mw", -8.0)
    return "under" if pred <= under else "over" if pred >= over else "near_zero"


@router.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    held = _excluded_in(req.sequence[0].date, req.target_date)
    if held:
        return PredictResponse(status="held", target_date=req.target_date, issued_at=_now(),
                               reason=f"입력 창에 품질 제외일이 있습니다: {held}")
    model = model_loader.get_model()
    pred = model.predict_one([p.model_dump() for p in req.sequence])
    issued = _now()
    state.record_prediction(req.target_date.isoformat(), pred, model.version, issued)
    return PredictResponse(status="ok", target_date=req.target_date, predicted_gap_mw=round(pred, 2),
                           direction=direction_of(pred), model_version=model.version, issued_at=issued)


@router.post("/predict/replay", response_model=ReplayResponse)
def replay(req: ReplayRequest):
    hist = [p.model_dump() for p in req.history]
    preds, events, retrains = [], [], []
    last_check: dict = {"status": "ok"}
    for i in range(SEQ_LEN, len(hist)):
        model = model_loader.get_model()  # 승격되면 캐시가 비워져 새 champion이 로드된다
        window = hist[i - SEQ_LEN:i]
        target = hist[i]
        t0 = time.perf_counter()
        pred = model.predict_one(window)
        reqlog.record("/predict (replay)", 200, (time.perf_counter() - t0) * 1000)
        d = target["date"].isoformat()
        state.record_prediction(d, pred, model.version, _now(), req.label)
        rec = state.record_actual(d, target["gap_mw"])
        preds.append({"date": d, "predicted": round(pred, 2), "actual": round(target["gap_mw"], 2),
                      "direction": direction_of(pred), "model_version": model.version})
        last_check = check_and_trigger(state.confirmed_records(upto=d), as_of=d)
        if last_check["status"] != "ok":
            ev = {k: last_check.get(k) for k in ("status", "date", "rolling_rmse_mw", "rolling_rmse_incl_special_mw",
                                                  "threshold_mw", "n_used", "consecutive", "breach_streak_days",
                                                  "breach_streak_start", "cooldown_days_left")}
            events.append(ev)
        if last_check["status"] == "retrain_triggered":
            retrains.append({"as_of": d, **{k: last_check.get(k) for k in ("promoted", "rmse", "champion_rmse", "version")},
                             "retrain": last_check.get("retrain")})
    state.last_replay_end = hist[-1]["date"].isoformat()
    summary = {k: last_check.get(k) for k in ("status", "rolling_rmse_mw", "rolling_rmse_incl_special_mw",
                                              "threshold_mw", "n_used", "n_records", "special_excluded",
                                              "breach_streak_days", "breach_streak_start")}
    return ReplayResponse(label=req.label, n_predictions=len(preds), predictions=preds, events=events,
                          retrains=retrains, drift_check=summary)


@router.get("/predict/next")
def predict_next(as_of: str | None = None):
    """대시보드용: 등록 이력에서 as_of(기본: 마지막 날)까지의 최신 N일로 다음 날 갭을 계산한다. 이력·요청 로그에 남기지 않는다."""
    from data.features import load_rows
    from data.storage import latest_upload
    try:
        rows = load_rows(latest_upload())
    except FileNotFoundError:
        return {"status": "no_data"}
    if as_of:
        rows = [r for r in rows if r["Date"] <= as_of]
    seq = rows[-SEQ_LEN:]
    if len(seq) < SEQ_LEN:
        return {"status": "no_data"}
    model = model_loader.get_model()
    pred = model.predict_one([{"gap_mw": r["GapMW"], "forecast_mw": r["ForecastMW"]} for r in seq])
    target = (dt.date.fromisoformat(seq[-1]["Date"]) + dt.timedelta(days=1)).isoformat()
    return {"status": "ok", "target_date": target, "predicted_gap_mw": round(pred, 2), "direction": direction_of(pred),
            "model_version": model.version, "based_on": [r["Date"] for r in seq]}


@router.post("/predict/reset")
def reset_state():
    """시연용: 예측 이력·누적 초과 일수·쿨다운 등 프로세스 운영 상태를 비운다 (레지스트리·모델은 그대로)."""
    state.reset()
    return {"status": "reset", "model_version": model_loader.current_version()}


@router.get("/predict/history")
def history(n: int = 60):
    return {"records": state.recent(n), "last_actual_date": state.last_actual_date, "last_replay_end": state.last_replay_end,
            "consecutive_breaches": state.consecutive_breaches, "last_retrain": state.last_retrain}
