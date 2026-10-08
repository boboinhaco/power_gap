"""
전처리 어댑터 (명세서 v2 3장): data/raw/ 원본 → 학습·서빙용 정규화 CSV 1개 + 품질 보고서.

출력
  data/jeju_gap.csv          : Date,GapMW,ForecastMW   (18~21시 평균, GapMW = 예측 − 실적)
  data/quality_report.md     : 파일별 인코딩·기간·중복·제외일과 사유
  data/excluded_dates.json   : 제외일 목록 (/predict가 "보류" 응답을 내는 근거)

처리 규칙
  1) 누적본 중복: 같은 날짜가 여러 파일에 있으면 허용 오차(0.01MW) 이내일 때 최신 파일 값 채택,
     초과하면 그 날짜를 제외하고 사유 기록
  2) 18~21시 네 값 중 하나라도 없으면 제외 (보간하지 않음)
  3) 예측·실적 중 한쪽만 있는 날짜는 조인에서 자연히 빠진다 (제외일로 기록)

실행: python data/prepare_power_gap.py [--raw data/raw] [--out data/jeju_gap.csv] [--seed-upload]
  --seed-upload : 생성한 CSV를 data/uploads/power_seed.csv로 복사 (Docker 빌드·로컬 시작용)
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.raw_readers import list_raw_files, read_actual_csv, read_forecast_xlsx

EVENING_HOURS = (18, 19, 20, 21)
DUP_TOLERANCE_MW = 0.01


def detect_encoding(path: str) -> str:
    raw = open(path, "rb").read()
    try:
        raw.decode("utf-8-sig"); return "utf-8-sig"
    except UnicodeDecodeError:
        return "cp949"


def merge_with_dedup(sources: list[tuple[str, dict]], report: list[str], excluded: dict[str, str]) -> dict:
    """
    [(파일명, {date: [h1..h24]})] → {date: [..]}.
    같은 날짜가 여러 파일에 있으면 **나중 파일(이름순 마지막 = 보통 다시 받은 파일) 값을 채택**한다.
    값 차이가 허용 오차(0.01MW)를 넘으면 제외하지 않고 보고서에 "갱신됨"으로 기록만 남긴다
    (같은 자료를 다시 내려받으면 예측치가 갱신돼 값이 달라지는 것이 정상이므로).
    """
    merged, origin = {}, {}
    n_dup, n_conflict, samples = 0, 0, []
    for name, table in sources:  # 정렬된 순서 = 오래된 파일 → 최신 파일
        for d, vals in table.items():
            if d in merged:
                n_dup += 1
                diffs = [abs(a - b) for a, b in zip(merged[d], vals) if a is not None and b is not None]
                if diffs and max(diffs) > DUP_TOLERANCE_MW:
                    n_conflict += 1
                    if len(samples) < 3:
                        samples.append(f"{d} {origin[d]}→{name} 최대 {max(diffs):.1f}MW")
            merged[d] = vals
            origin[d] = name
    report.append(f"- 중복 날짜 {n_dup}건, 그중 값이 달라 최신 파일로 갱신한 날 {n_conflict}건"
                  + (f" (예: {'; '.join(samples)})" if samples else ""))
    return merged


def evening_mean(vals: list[float | None]) -> float | None:
    picked = [vals[h - 1] for h in EVENING_HOURS]
    if any(v is None for v in picked):
        return None
    return sum(picked) / len(picked)


def build_gap_dataset(raw_dir: str, info: dict | None = None) -> tuple[list[tuple[str, float, float]], list[str], dict[str, str]]:
    """raw_dir 안의 제주 예측 XLSX·실적 CSV 전부를 조인해 (rows, report_lines, excluded)를 돌려준다.
    rows = [(date, gap_mw, forecast_mw)] 날짜순. 서버(/data/upload-raw)와 CLI가 같이 쓴다."""
    report = ["# 데이터 품질 보고서 (data/prepare_power_gap.py)", "", "## 입력 파일", ""]
    excluded: dict[str, str] = {}
    fc_sources, ac_sources = [], []
    for nf, path in list_raw_files(raw_dir):
        if nf.lower().startswith("pssjeju") and nf.endswith(".xlsx"):
            try:
                t = read_forecast_xlsx(path)
            except Exception as e:  # 깨진 파일 등
                t = {}; report.append(f"- 예측 `{nf}`: 읽기 실패 ({e})"); continue
            if not t:
                report.append(f"- 예측 `{nf}`: ⚠ 날짜 행을 읽지 못해 건너뜀 (헤더 `구분/1h…24h`와 `YYYYMMDD` 날짜 형식을 확인)"); continue
            fc_sources.append((nf, t))
            report.append(f"- 예측 `{nf}`: {len(t)}일 {min(t)} ~ {max(t)}")
        elif nf.endswith(".csv") and "제주전력수요" in nf:
            t = read_actual_csv(path)
            if not t:
                report.append(f"- 실적 `{nf}`: ⚠ 날짜 행을 읽지 못해 건너뜀"); continue
            ac_sources.append((nf, t))
            report.append(f"- 실적 `{nf}`: {len(t)}일 {min(t)} ~ {max(t)}, 인코딩 {detect_encoding(path)}")
        else:
            report.append(f"- 건너뜀 `{nf}` (제주 예측/실적 아님)")
    if not fc_sources or not ac_sources:
        raise ValueError("제주 예측 XLSX(pssJeju_*.xlsx)와 실적 CSV(…제주전력수요_*.csv)가 모두 있어야 갭을 계산할 수 있습니다")

    report += ["", "## 병합"]
    forecast = merge_with_dedup(fc_sources, report, excluded)
    actual = merge_with_dedup(ac_sources, report, excluded)

    rows = []
    only_fc = sorted(set(forecast) - set(actual))
    only_ac = sorted(set(actual) - set(forecast))
    for d in sorted(set(forecast) & set(actual)):
        if d in excluded:
            continue
        f, a = evening_mean(forecast[d]), evening_mean(actual[d])
        if f is None or a is None:
            excluded[d] = "18~21시 값 누락"; continue
        if f <= 0:
            excluded[d] = "ForecastMW <= 0"; continue
        rows.append((d, f - a, f))
    if not rows:
        raise ValueError("예측과 실적이 겹치는 날짜가 없습니다")
    if info is not None:
        info["actual_only"] = {"n": len(only_ac), "start": only_ac[0] if only_ac else None, "end": only_ac[-1] if only_ac else None}
        info["forecast_only"] = {"n": len(only_fc), "start": only_fc[0] if only_fc else None, "end": only_fc[-1] if only_fc else None}
        info["joined"] = {"n": len(rows), "start": rows[0][0], "end": rows[-1][0]}
        after = [d for d in only_ac if d > rows[-1][0]]
        info["actual_only_after"] = {"n": len(after), "start": after[0] if after else None, "end": after[-1] if after else None}

    import datetime as dt
    gaps = [g for _, g, _ in rows]
    mean = sum(gaps) / len(gaps)
    std = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    span = (dt.date.fromisoformat(rows[-1][0]) - dt.date.fromisoformat(rows[0][0])).days + 1
    report += ["", "## 결과", "",
               f"- 조인 결과 {len(rows)}일 {rows[0][0]} ~ {rows[-1][0]} (달력 {span}일, 결측 {span - len(rows)}일)",
               f"- GapMW 평균 {mean:.1f} / 표준편차 {std:.1f} / 최소 {min(gaps):.1f} / 최대 {max(gaps):.1f}",
               f"- 과소예측(Gap<0) 비율 {sum(g < 0 for g in gaps) / len(gaps):.1%}",
               f"- 예측만 있는 날 {len(only_fc)}일, 실적만 있는 날 {len(only_ac)}일 (조인 제외)"
               + (f": 실적만 {only_ac[0]} ~ {only_ac[-1]}" if only_ac else "")
               + (f", 예측만 {only_fc[0]} ~ {only_fc[-1]}" if only_fc else ""),
               "", "## 제외일", ""]
    report += [f"- {d}: {why}" for d, why in sorted(excluded.items())] or ["- 없음"]
    return rows, report, excluded


def write_outputs(rows, report, excluded, out_csv: str, info: dict | None = None) -> None:
    out_dir = os.path.dirname(out_csv) or "."
    os.makedirs(out_dir, exist_ok=True)
    if info is not None:
        with open(os.path.join(out_dir, "unmatched.json"), "w", encoding="utf-8") as fp:
            json.dump(info, fp, ensure_ascii=False, indent=2)
    with open(out_csv, "w", encoding="utf-8", newline="") as fp:
        fp.write("Date,GapMW,ForecastMW\n")
        for d, g, f in rows:
            fp.write(f"{d},{g:.3f},{f:.3f}\n")
    with open(os.path.join(out_dir, "excluded_dates.json"), "w", encoding="utf-8") as fp:
        json.dump(excluded, fp, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "quality_report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(report) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/jeju_gap.csv")
    ap.add_argument("--seed-upload", action="store_true")
    args = ap.parse_args()
    info: dict = {}
    try:
        rows, report, excluded = build_gap_dataset(args.raw, info)
    except ValueError as e:
        sys.exit(str(e))
    write_outputs(rows, report, excluded, args.out, info)
    print("\n".join(report))
    print(f"- 출력 `{args.out}`")
    if args.seed_upload:
        os.makedirs("data/uploads", exist_ok=True)
        dest = "data/uploads/power_seed.csv"
        shutil.copyfile(args.out, dest)
        print(f"seeded -> {dest}")


if __name__ == "__main__":
    main()
