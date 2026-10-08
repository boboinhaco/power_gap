"""
요청 로그 (운영 지표용). 각 API 요청과 재생 중 하루치 예측을 logs/requests.log 에 한 줄씩 기록하고,
대시보드의 "운영 지표 요약"이 시간 구간(5분/1시간/6시간/24시간)별로 집계해 보여준다.
형식: <unix_ts>,<path>,<status>,<latency_ms>
"""
import os
import threading
import time

LOG_PATH = os.getenv("REQUESTS_LOG", "logs/requests.log")
_lock = threading.Lock()


def record(path: str, status: int, latency_ms: float) -> None:
    os.makedirs(os.path.dirname(LOG_PATH) or ".", exist_ok=True)
    with _lock, open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"{time.time():.3f},{path},{status},{latency_ms:.1f}\n")


def _read(since: float) -> list[tuple[float, str, int, float]]:
    if not os.path.exists(LOG_PATH):
        return []
    out = []
    with open(LOG_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                ts, path, status, ms = line.rstrip("\n").split(",")
                ts = float(ts)
            except ValueError:
                continue
            if ts >= since:
                out.append((ts, path, int(status), float(ms)))
    return out


def summary(window_seconds: int, buckets: int = 12) -> dict:
    now = time.time()
    rows = _read(now - window_seconds)
    n = len(rows)
    width = window_seconds / buckets
    counts = [0] * buckets
    lat_sum = [0.0] * buckets
    for ts, _, _, ms in rows:
        i = min(buckets - 1, int((ts - (now - window_seconds)) / width))
        counts[i] += 1
        lat_sum[i] += ms
    return {
        "window_seconds": window_seconds,
        "requests": n,
        "avg_latency_ms": (sum(r[3] for r in rows) / n) if n else 0.0,
        "success_rate": (sum(1 for r in rows if r[2] < 400) / n) if n else None,
        "series": {"requests": counts, "latency_ms": [lat_sum[i] / counts[i] if counts[i] else 0.0 for i in range(buckets)]},
        "by_path": {p: sum(1 for r in rows if r[1] == p) for p in sorted({r[1] for r in rows})},
    }
