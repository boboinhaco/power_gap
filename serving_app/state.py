"""
서빙 프로세스의 운영 상태 (명세서 v2 5장 "상태 저장과 실행 모델").

MVP는 **단일 워커** 전제다. 예측 이력은 메모리에 날짜 키로 들고, 실적이 연결될 때마다
data/predictions.csv 에 한 줄씩 append 해서 재시작 후에도 추적할 수 있게 한다.
같은 날짜는 중복 등록하지 않는다 (가장 최근 예측으로 갱신).
"""
import csv
import os
import threading

PREDICTIONS_CSV = os.getenv("PREDICTIONS_CSV", "data/predictions.csv")
_FIELDS = ["target_date", "predicted_gap_mw", "actual_gap_mw", "model_version", "issued_at", "label"]

_lock = threading.Lock()
_records: dict[str, dict] = {}
consecutive_breaches = 0  # 재학습 판정용 (쿨다운·재학습 후 리셋)
breach_streak_days = 0    # 임계값 초과가 며칠째 이어지는지 (아래로 내려오면 0) — 누적 경보 표시용
breach_streak_start: str | None = None
last_actual_date: str | None = None
last_replay_end: str | None = None   # 가장 최근에 재생한 구간의 마지막 실적일 = 대시보드 "기준일"
last_retrain: dict | None = None


def record_prediction(date: str, predicted: float, model_version: str, issued_at: str, label: str = "real") -> None:
    with _lock:
        prev = _records.get(date, {})
        _records[date] = {"date": date, "predicted": predicted, "actual": prev.get("actual"),
                          "model_version": model_version, "issued_at": issued_at, "label": label}


def record_actual(date: str, actual: float) -> dict | None:
    global last_actual_date
    with _lock:
        r = _records.get(date)
        if r is None:
            return None
        r["actual"] = actual
        if last_actual_date is None or date > last_actual_date:
            last_actual_date = date
        _append_csv(r)
        return r


def confirmed_records(upto: str | None = None) -> list[dict]:
    """확정 라벨(실적 연결됨) 기록을 날짜순으로. upto를 주면 그 날짜까지만 (이력 재생 시 "그 시점"의 창을 보기 위해)."""
    with _lock:
        return [dict(r) for d, r in sorted(_records.items())
                if r.get("actual") is not None and (upto is None or d <= upto)]


def recent(n: int = 50) -> list[dict]:
    with _lock:
        return [dict(r) for d, r in sorted(_records.items())][-n:]


def reset() -> None:
    global consecutive_breaches, last_actual_date, last_retrain, breach_streak_days, breach_streak_start, last_replay_end
    with _lock:
        _records.clear()
    consecutive_breaches = 0
    last_replay_end = None
    breach_streak_days = 0
    breach_streak_start = None
    last_actual_date = None
    last_retrain = None


def _append_csv(r: dict) -> None:
    os.makedirs(os.path.dirname(PREDICTIONS_CSV) or ".", exist_ok=True)
    new = not os.path.exists(PREDICTIONS_CSV)
    with open(PREDICTIONS_CSV, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        if new:
            w.writeheader()
        w.writerow({"target_date": r["date"], "predicted_gap_mw": round(r["predicted"], 3),
                    "actual_gap_mw": round(r["actual"], 3), "model_version": r["model_version"],
                    "issued_at": r["issued_at"], "label": r["label"]})
