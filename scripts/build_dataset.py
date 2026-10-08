"""
data/raw/ 안의 전력거래소 원본 파일(수요예측 xlsx, 실적 csv)을 하나의 long-format CSV로 병합한다.

출력: data/power_demand_merged.csv
  datetime     : 시간대 시작 시각 (hour 라벨 h는 (h-1):00~h:00 구간. 24시 = 23:00~24:00)
  date, hour   : 원본 날짜와 1~24 시간 라벨
  region       : jeju | inland | national
  forecast_mw  : 하루전 발전계획용 수요예측 (KPX). national = inland + jeju 합산(파생)
  actual_mw    : 시간별 전력수요량 실적 (공공데이터포털). inland = national - jeju (파생)
  error_mw     : forecast_mw - actual_mw  (음수 = 과소예측)
  error_pct    : error_mw / actual_mw * 100
  forecast_src, actual_src : 값이 원본(raw)인지 파생(derived)인지

표준 라이브러리만 사용한다 (openpyxl 불필요). 원본 리더는 data/raw_readers.py 공용.
학습·서빙용 일별 CSV(Date,GapMW,ForecastMW)는 data/prepare_power_gap.py가 만든다.
"""
import csv, os, sys, unicodedata, datetime as dt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "data", "power_demand_merged.csv")
DAILY_OUT = {"jeju": os.path.join(ROOT, "data", "jeju_gap_daily.csv"),
             "national": os.path.join(ROOT, "data", "national_gap_daily.csv")}
EVENING_HOURS = (18, 19, 20, 21)   # PDF 2번: 다음날 18~21시 평균 (시간 라벨 기준)


from data.raw_readers import read_actual_csv, read_forecast_xlsx, to_float  # noqa: E402  (공용 리더)


def main():
    forecast = {"jeju": {}, "inland": {}}
    actual = {"jeju": {}, "national": {}}
    files = sorted(os.listdir(RAW))
    for f in files:
        nf = unicodedata.normalize("NFC", f)
        p = os.path.join(RAW, f)
        if nf.lower().startswith("pssjeju") and nf.endswith(".xlsx"):
            forecast["jeju"].update(read_forecast_xlsx(p))
        elif nf.lower().startswith("pssinland") and nf.endswith(".xlsx"):
            forecast["inland"].update(read_forecast_xlsx(p))
        elif nf.endswith(".csv") and "제주전력수요" in nf:
            actual["jeju"].update(read_actual_csv(p))
        elif nf.endswith(".csv") and "전국 전력수요량" in nf:
            actual["national"].update(read_actual_csv(p))
        else:
            print("skip:", nf, file=sys.stderr)

    # 파생: national forecast = inland + jeju, inland actual = national - jeju
    forecast["national"] = {}
    for d in set(forecast["inland"]) & set(forecast["jeju"]):
        forecast["national"][d] = [a + b if a is not None and b is not None else None
                                   for a, b in zip(forecast["inland"][d], forecast["jeju"][d])]
    actual["inland"] = {}
    for d in set(actual["national"]) & set(actual["jeju"]):
        actual["inland"][d] = [a - b if a is not None and b is not None else None
                               for a, b in zip(actual["national"][d], actual["jeju"][d])]
    src = {("forecast", "national"): "derived", ("actual", "inland"): "derived"}

    rows = []
    for region in ("jeju", "inland", "national"):
        dates = sorted(set(forecast.get(region, {})) | set(actual.get(region, {})))
        for d in dates:
            fc = forecast.get(region, {}).get(d)
            ac = actual.get(region, {}).get(d)
            base = dt.datetime.fromisoformat(d)
            for h in range(1, 25):
                f = fc[h - 1] if fc else None
                a = ac[h - 1] if ac else None
                if f is None and a is None:
                    continue
                err = f - a if (f is not None and a is not None) else None
                pct = err / a * 100 if (err is not None and a) else None
                rows.append({
                    "datetime": (base + dt.timedelta(hours=h - 1)).strftime("%Y-%m-%d %H:%M"),
                    "date": d, "hour": h, "region": region,
                    "forecast_mw": "" if f is None else f"{f:.3f}",
                    "actual_mw": "" if a is None else f"{a:.3f}",
                    "error_mw": "" if err is None else f"{err:.3f}",
                    "error_pct": "" if pct is None else f"{pct:.3f}",
                    "forecast_src": "" if f is None else src.get(("forecast", region), "raw"),
                    "actual_src": "" if a is None else src.get(("actual", region), "raw"),
                })
    rows.sort(key=lambda r: (r["region"], r["datetime"]))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 스켈레톤 호환 일별 CSV (Date,Open,High,Low,Close,Volume)
    for region, path in DAILY_OUT.items():
        daily = []
        for d in sorted(set(forecast.get(region, {})) & set(actual.get(region, {}))):
            fc, ac = forecast[region][d], actual[region][d]
            errs = [fc[h - 1] - ac[h - 1] for h in EVENING_HOURS
                    if fc[h - 1] is not None and ac[h - 1] is not None]
            fcs = [fc[h - 1] for h in EVENING_HOURS if fc[h - 1] is not None]
            if len(errs) < len(EVENING_HOURS):
                continue
            daily.append({"Date": d,
                          "Open": f"{errs[0]:.2f}", "High": f"{max(errs):.2f}", "Low": f"{min(errs):.2f}",
                          "Close": f"{sum(errs) / len(errs):.2f}",
                          "Volume": f"{sum(fcs) / len(fcs):.2f}"})
        with open(path, "w", newline="", encoding="utf-8") as fp:
            w = csv.DictWriter(fp, fieldnames=["Date", "Open", "High", "Low", "Close", "Volume"])
            w.writeheader(); w.writerows(daily)
        print(f"wrote {path}  rows={len(daily)}  {daily[0]['Date']} ~ {daily[-1]['Date']}")

    # 요약
    print(f"wrote {OUT}  rows={len(rows):,}")
    for region in ("jeju", "inland", "national"):
        rr = [r for r in rows if r["region"] == region]
        fcd = sorted({r["date"] for r in rr if r["forecast_mw"]})
        acd = sorted({r["date"] for r in rr if r["actual_mw"]})
        both = sorted({r["date"] for r in rr if r["error_mw"]})
        def rng(x): return f"{x[0]} ~ {x[-1]} ({len(x)}일)" if x else "-"
        print(f"[{region:8}] forecast {rng(fcd)} | actual {rng(acd)} | error 계산 가능 {rng(both)}")


if __name__ == "__main__":
    main()
