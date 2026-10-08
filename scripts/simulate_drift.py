"""
Day3 드리프트 시연 (명세서 v2 5장 "이력 재생 배치").

주가 랜덤워크 대신 **실제 전력 갭 이력**을 날짜순으로 서버에 재생한다.
기본 구간 2025-09-01~10-31에는 추석·개천절 연휴가 들어 있어 합성 없이도 경보 흐름을 볼 수 있다.
--synthetic 을 주면 재생 구간의 실제 갭을 -60MW 이동한 "합성 시나리오"를 보내며, 응답·로그에 synthetic 라벨이 붙는다.

    python scripts/simulate_drift.py                       # 로컬 uvicorn (8077), 실제 이력
    python scripts/simulate_drift.py --target container    # 도커 컨테이너 (8099)
    python scripts/simulate_drift.py --start 2025-09-01 --end 2025-10-31 --synthetic
"""
import argparse
import datetime as dt
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.features import SEQ_LEN, load_rows, rows_between
from data.storage import latest_upload

TARGETS = {"local": "http://localhost:8077", "container": "http://localhost:8099"}
SYNTHETIC_SHIFT_MW = -60.0


def build_history(start: str, end: str, synthetic: bool) -> list[dict]:
    rows = load_rows(latest_upload())
    lead = (dt.date.fromisoformat(start) - dt.timedelta(days=SEQ_LEN)).isoformat()
    hist = [{"date": r["Date"], "gap_mw": r["GapMW"], "forecast_mw": r["ForecastMW"]} for r in rows_between(rows, lead, end)]
    if synthetic:
        hist = [{**p, "gap_mw": p["gap_mw"] + SYNTHETIC_SHIFT_MW} if p["date"] >= start else p for p in hist]
    return hist


def main():
    ap = argparse.ArgumentParser(description="제주 갭 이력 재생 드리프트 시연")
    ap.add_argument("--target", choices=["local", "container", "both"], default="local")
    ap.add_argument("--start", default="2025-09-01"); ap.add_argument("--end", default="2025-10-31")
    ap.add_argument("--synthetic", action="store_true", help="실제 갭을 -60MW 이동한 합성 시나리오 (라벨 표시)")
    ap.add_argument("--url", help="서버 주소 직접 지정 (예: http://localhost:8078)")
    args = ap.parse_args()
    label = "synthetic" if args.synthetic else "real"
    hist = build_history(args.start, args.end, args.synthetic)
    print(f"[1] 이력 {len(hist)}일 ({hist[0]['date']} ~ {hist[-1]['date']}), 라벨={label}")

    if args.url:
        TARGETS["custom"] = args.url.rstrip("/")
    for name in (["custom"] if args.url else ["local", "container"] if args.target == "both" else [args.target]):
        url = f"{TARGETS[name]}/predict/replay"
        print(f"\n=== {name} ({url}) ===")
        try:
            resp = requests.post(url, json={"history": hist, "label": label}, timeout=600)
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            print(f"[skip] {TARGETS[name]} 연결 실패 - 서버(/health)를 확인하세요"); continue
        r = resp.json()
        dc = r["drift_check"]
        print(f"[2] {r['n_predictions']}일 재생. 마지막 롤링 RMSE {dc['rolling_rmse_mw']:.1f} MW "
              f"(특수일 포함 {dc['rolling_rmse_incl_special_mw']:.1f}) / 임계값 {dc['threshold_mw']:.1f}")
        for e in r["events"]:
            print(f"    {e['date']} {e['status']:18s} rmse={e['rolling_rmse_mw']:.1f} thr={e['threshold_mw']:.1f}")
        for t in r["retrains"]:
            print(f"[3] {t['as_of']} 재학습: promoted={t['promoted']} candidate={t.get('rmse')} champion={t.get('champion_rmse')} "
                  f"version={t.get('version')} ({(t.get('retrain') or {}).get('status')})")
        if not r["retrains"]:
            print("[3] 재학습 트리거 없음 (연속 2회 초과 조건 미충족)")
    print("\n[4] 대시보드의 재학습 로그 또는 logs/aiops.log에서 [WATCH]/[WARN]/[INFO]/[OK] 순서를 확인하세요.")


if __name__ == "__main__":
    main()
