"""
[Day3] 드리프트 감지 — serving_app/monitoring/drift_detector.py (명세서 v2 5장)

최근 WINDOW_SIZE(21)건의 확정 라벨로 "우리 갭 예측이 실제 갭과 평균 몇 MW 틀리는지"(RMSE)를 본다.
  - 임계값: 학습 시 thresholds.json에 저장된 2024년 롤링 RMSE 95분위 (특수일 제외). 실습의 $4는 쓰지 않는다
  - 특수일(설·추석 연휴 등, data/holidays.py)은 창에서 제외하고 계산한다. 제외 전·후 RMSE를 둘 다 돌려준다
  - 임계값 초과가 연속 CONSECUTIVE_REQUIRED(2)회일 때만 재학습 (한 번의 특수일로 재학습하지 않음)
"""
import math
import os

from data.holidays import special_days
from data.metrics import bias, mae, rolling_rmse, wape
from serving_app.evaluation import load_thresholds

WINDOW_SIZE = 21
CONSECUTIVE_REQUIRED = 2
MIN_USED = WINDOW_SIZE // 2  # 특수일 제외 후 남는 최소 건수
FALLBACK_THRESHOLD = float(os.getenv("DRIFT_RMSE_THRESHOLD", "55.0"))


def threshold_mw() -> float:
    th = load_thresholds()
    return float(th["drift_rmse_threshold_mw"]) if th else FALLBACK_THRESHOLD


def compute_rmse(recent_predictions: list[dict]) -> float:
    if not recent_predictions:
        return 0.0
    return math.sqrt(sum((p["actual"] - p["predicted"]) ** 2 for p in recent_predictions) / len(recent_predictions))


def evaluate(records: list[dict]) -> dict:
    """records: 날짜순 [{date, predicted, actual}] (확정 라벨만). 마지막 시점의 판정 정보를 돌려준다."""
    if not records:
        return {"n_records": 0, "ready": False, "breach": False, "threshold_mw": threshold_mw()}
    window = records[-WINDOW_SIZE:]
    excl = {d.isoformat() for d in special_days(window[0]["date"], window[-1]["date"])}
    used = [r for r in window if r["date"] not in excl]
    rmse_all = compute_rmse(window)
    rmse_excl = compute_rmse(used)
    yt, yp = [r["actual"] for r in used], [r["predicted"] for r in used]
    th = threshold_mw()
    ready = len(records) >= WINDOW_SIZE and len(used) >= MIN_USED
    return {
        "date": records[-1]["date"], "n_records": len(records), "ready": ready,
        "rolling_rmse_mw": rmse_excl, "rolling_rmse_incl_special_mw": rmse_all,
        # 보조 지표(판정에는 쓰지 않음): MAE, WAPE(=Σ|잔차|/Σ|실제 갭|, naive=1.0), Bias(=mean(예측−실제))
        "rolling_mae_mw": mae(yt, yp), "rolling_wape": wape(yt, yp), "rolling_bias_mw": bias(yt, yp),
        "n_used": len(used), "special_excluded": sorted(d for d in excl if any(r["date"] == d for r in window)),
        "threshold_mw": th, "breach": bool(ready and rmse_excl > th),
    }


def is_drift(records: list[dict]) -> bool:
    return evaluate(records)["breach"]
