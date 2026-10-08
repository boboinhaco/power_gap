"""
[Day3] 드리프트 → 자동 재학습 — serving_app/monitoring/retrain_trigger.py (명세서 v2 5장)

흐름
  롤링 RMSE(특수일 제외) > 임계값 → 연속 2회째인가?
    아니오 → [WATCH] 로그만
    예     → [WARN] → [INFO] retrain triggered (train=[D-90,D-15], val=[D-14,D-1])
            → champion 가중치 fine-tuning → 같은 검증 14일에서 후보 vs champion
            → 후보가 낮으면 alias 승격 + 서빙 캐시 무효화 + [OK] 로그, 아니면 [INFO] gate failed (기존 유지)
쿨다운: 한 번 재학습(승격/차단 무관)한 뒤 COOLDOWN_DAYS(14)일 동안은 다시 재학습하지 않는다.
  롤링 창이 임계값 위에 머무는 동안 이틀마다 재학습이 반복되는 것을 막고, 다음 후보의 검증 창 [D-14, D-1]이
  직전 재학습의 검증 창과 겹치지 않게 한다 (2025-09~10 실제 재생에서 61일에 12회 재학습이 나와 추가).
로그 문장의 앞부분([WARN]/[INFO]/[OK])은 대시보드가 읽으므로 바꾸지 않는다.
"""
import datetime as dt
import logging
import os

from serving_app import model_loader, state
from serving_app.monitoring.drift_detector import CONSECUTIVE_REQUIRED, evaluate

logger = logging.getLogger("aiops")
COOLDOWN_DAYS = 14


def _in_cooldown(as_of: str) -> int:
    """직전 재학습 이후 경과일이 COOLDOWN_DAYS 미만이면 남은 일수, 아니면 0."""
    last = (state.last_retrain or {}).get("as_of")
    if not last or not as_of:
        return 0
    elapsed = (dt.date.fromisoformat(as_of) - dt.date.fromisoformat(last)).days
    if elapsed < 0:  # 직전 재학습보다 과거 날짜를 재생 중 → 재생 시점에는 그 재학습이 아직 없다
        return 0
    return COOLDOWN_DAYS - elapsed if elapsed < COOLDOWN_DAYS else 0


REFERENCE_CSV = os.getenv("REFERENCE_CSV", "data/jeju_gap.csv")  # 업로드 이력이 짧을 때 이전 최신 데이터를 붙여 오는 기준 파일


def rows_for_retrain(as_of: str) -> tuple[list[dict], dict]:
    """
    재학습에 쓸 행. 최신 업로드 파일을 기본으로 하되, [D-(RETRAIN_ROWS+SEQ_LEN), D-1] 구간에 빠진 날짜가 있으면
    REFERENCE_CSV(2024~2025 정규화 이력)에서 그 날짜를 붙인다. 예: 2026-01에 일찍 감지되면 2025-11~12월이 붙는다.
    """
    from data.features import SEQ_LEN, load_rows, rows_between
    from data.storage import latest_upload
    from serving_app.train_and_register import RETRAIN_ROWS
    D = dt.date.fromisoformat(as_of)
    need_start = (D - dt.timedelta(days=RETRAIN_ROWS + SEQ_LEN)).isoformat()
    rows = load_rows(latest_upload())
    have = {r["Date"] for r in rows}
    info = {"backfilled": 0, "backfill_range": None, "source": os.path.basename(latest_upload())}
    missing = [d for d in (need_start <= x <= (D - dt.timedelta(days=1)).isoformat() and x for x in
               [(dt.date.fromisoformat(need_start) + dt.timedelta(days=i)).isoformat()
                for i in range(RETRAIN_ROWS + SEQ_LEN)]) if d and d not in have]
    if missing and os.path.exists(REFERENCE_CSV):
        ref = {r["Date"]: r for r in load_rows(REFERENCE_CSV)}
        added = [ref[d] for d in missing if d in ref]
        if added:
            rows = sorted(rows + added, key=lambda r: r["Date"])
            info.update({"backfilled": len(added), "backfill_range": [added[0]["Date"], added[-1]["Date"]]})
    return rows, info


def _update_streak(info: dict, as_of: str) -> None:
    if info["breach"]:
        if state.breach_streak_days == 0:
            state.breach_streak_start = as_of
        state.breach_streak_days += 1
    else:
        state.breach_streak_days = 0
        state.breach_streak_start = None
    info["breach_streak_days"] = state.breach_streak_days
    info["breach_streak_start"] = state.breach_streak_start


def check_and_trigger(records: list[dict], as_of: str | None = None) -> dict:
    info = evaluate(records)
    as_of = as_of or info.get("date")
    _update_streak(info, as_of)
    if not info["breach"]:
        state.consecutive_breaches = 0
        return {"status": "ok", **info}

    remaining = _in_cooldown(as_of)
    if remaining:
        logger.warning(f"[WATCH] rolling RMSE {info['rolling_rmse_mw']:.1f} MW > {info['threshold_mw']:.1f} MW "
                       f"(누적 {info['breach_streak_days']}일째) but cooldown ({remaining}d left after retrain at "
                       f"{state.last_retrain['as_of']}, as_of={as_of})")
        return {"status": "cooldown", "cooldown_days_left": remaining, **info}

    state.consecutive_breaches += 1
    if state.consecutive_breaches < CONSECUTIVE_REQUIRED:
        logger.warning(f"[WATCH] rolling RMSE {info['rolling_rmse_mw']:.1f} MW > {info['threshold_mw']:.1f} MW "
                       f"(누적 {info['breach_streak_days']}일째, {state.consecutive_breaches}/{CONSECUTIVE_REQUIRED}, as_of={as_of}, "
                       f"incl_special={info['rolling_rmse_incl_special_mw']:.1f})")
        return {"status": "watch", "consecutive": state.consecutive_breaches, **info}

    logger.warning(f"[WARN] drift detected - triggering retrain (rolling RMSE {info['rolling_rmse_mw']:.1f} MW "
                   f"> {info['threshold_mw']:.1f} MW, 누적 {info['breach_streak_days']}일째 "
                   f"(since {info['breach_streak_start']}), as_of={as_of})")
    from serving_app.train_and_register import fine_tune

    state.consecutive_breaches = 0
    rows, src = rows_for_retrain(as_of)
    result = fine_tune(rows, as_of)
    result["data_source"] = src
    bf = f", backfilled {src['backfilled']} rows {src['backfill_range'][0]}~{src['backfill_range'][1]} from reference" if src["backfilled"] else ""
    logger.info(f"[INFO] retrain triggered (train={result['train'][0]}~{result['train'][1]}, "
                f"val={result['val'][0]}~{result['val'][1]}, rows={result.get('n_train_rows', 0)}+{result.get('n_val', 0)}{bf})")
    state.last_retrain = {"as_of": as_of, **result}

    if result.get("promoted"):
        model_loader.invalidate_cache()
        logger.info(f"[OK] new_rmse={result['rmse']:.2f} (champion {result['champion_rmse']:.2f}, "
                    f"{result.get('gain_pct', 0):+.1f}% > min {result.get('min_gain_pct', 0):.0f}%) - "
                    f"production promoted: JejuGapPredictor v{result['version']}")
    elif "rmse" in result:
        logger.info(f"[INFO] gate failed: candidate {result['rmse']:.2f} vs champion {result['champion_rmse']:.2f} MW "
                    f"({result.get('gain_pct', 0):+.1f}%, min {result.get('min_gain_pct', 0):.0f}%, n_val={result['n_val']}) "
                    f"- keeping current champion")
    else:
        logger.info(f"[INFO] retrain held: {result.get('status')}")
    return {"status": "retrain_triggered", "promoted": bool(result.get("promoted")),
            "rmse": result.get("rmse"), "champion_rmse": result.get("champion_rmse"),
            "version": result.get("version"), "retrain": result, **info}
