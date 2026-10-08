"""
전력 갭 데이터 업로드·조회 — serving_app/routers/data.py

/data/upload     : 정규화 CSV(Date,GapMW,ForecastMW)를 받는다.
/data/upload-raw : 원본 두 종류(KPX 제주 예측 pssJeju_*.xlsx, 전력거래소 …제주전력수요_*.csv)를 여러 개 받아
                   data/raw/에 저장하고, 서버 안에서 전처리 어댑터(data/prepare_power_gap.py)를 돌려
                   기존 원본과 합친 정규화 CSV를 만들어 업로드 파일로 등록한다. 품질 보고서를 함께 돌려준다.
               컬럼·중복·비수치·정렬을 검증하고 data/uploads/power_<ts>.csv 로 저장한다.
/data/status : 최신 업로드 파일의 기간·갭 분포
/data/history: 이력 재생용 구간 조회 (lead 일수만큼 앞선 날짜를 포함)
"""
import datetime as dt
import os
import shutil
import tempfile
import time
import unicodedata

from fastapi import APIRouter, File, HTTPException, UploadFile

from data.features import SEQ_LEN, load_rows, rows_between
from data.prepare_power_gap import build_gap_dataset, write_outputs
from data.storage import UPLOAD_DIR, UPLOAD_PREFIX, latest_upload
from serving_app.monitoring.drift_detector import WINDOW_SIZE

router = APIRouter(prefix="/data")
MIN_ROWS = SEQ_LEN + WINDOW_SIZE
RAW_DIR = os.getenv("RAW_DIR", "data/raw")
NORMALIZED_CSV = "data/jeju_gap.csv"


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "UTF-8로 인코딩된 CSV만 업로드할 수 있습니다 (원본은 data/prepare_power_gap.py로 변환).")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        tmp.write(text)
    try:
        rows = load_rows(tmp.name)
    except ValueError as e:
        raise HTTPException(400, str(e))
    finally:
        os.unlink(tmp.name)
    if len(rows) < MIN_ROWS:
        raise HTTPException(400, f"최소 {MIN_ROWS}행 이상의 데이터가 필요합니다 (현재 {len(rows)}행)")

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"{UPLOAD_PREFIX}{int(time.time())}.csv")
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write("Date,GapMW,ForecastMW\n")
        for r in rows:
            f.write(f"{r['Date']},{r['GapMW']:.3f},{r['ForecastMW']:.3f}\n")
    return {"filename": os.path.basename(dest), "rows": len(rows), "start_date": rows[0]["Date"], "end_date": rows[-1]["Date"]}


def _save_rows(rows_iter, dest: str) -> None:
    with open(dest, "w", encoding="utf-8", newline="") as f:
        f.write("Date,GapMW,ForecastMW\n")
        for d, g, fc in rows_iter:
            f.write(f"{d},{g:.3f},{fc:.3f}\n")


@router.post("/upload-raw")
async def upload_raw(files: list[UploadFile] = File(...)):
    """원본 XLSX·CSV를 받아 서버에서 전처리한다. 파일명으로 종류를 판별한다 (pssJeju*.xlsx / *제주전력수요*.csv)."""
    os.makedirs(RAW_DIR, exist_ok=True)
    saved, rejected = [], []
    for f in files:
        name = unicodedata.normalize("NFC", os.path.basename(f.filename or ""))
        kind = ("forecast" if name.lower().startswith("pssjeju") and name.endswith(".xlsx")
                else "actual" if name.endswith(".csv") and "제주전력수요" in name else None)
        if kind is None:
            rejected.append({"file": name, "reason": "pssJeju_*.xlsx 또는 …제주전력수요_*.csv 만 받습니다"}); continue
        with open(os.path.join(RAW_DIR, name), "wb") as out:
            out.write(await f.read())
        saved.append({"file": name, "kind": kind})
    if not saved:
        raise HTTPException(400, {"detail": "처리할 수 있는 원본 파일이 없습니다", "rejected": rejected})
    info: dict = {}
    try:
        rows, report, excluded = build_gap_dataset(RAW_DIR, info)
    except ValueError as e:
        raise HTTPException(400, str(e))
    write_outputs(rows, report, excluded, NORMALIZED_CSV, info)
    warnings = []
    ao, fo = info.get("actual_only_after", {}), info.get("forecast_only", {})
    if ao.get("n"):
        warnings.append(f"실적만 있는 날 {ao['n']}일 ({ao['start']} ~ {ao['end']}): 같은 날짜의 KPX 예측이 없어 갭을 만들지 못했습니다 (pssJeju_{ao['end'][:4]}.xlsx 가 없거나, 있어도 그 날짜 행이 빠져 있음). 해당 기간이 포함된 예측 XLSX를 올리면 자동으로 합쳐집니다.")
    if fo.get("n"):
        warnings.append(f"예측만 있는 날 {fo['n']}일 ({fo['start']} ~ {fo['end']}): 실적 CSV가 없어 갭을 만들지 못했습니다.")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest = os.path.join(UPLOAD_DIR, f"{UPLOAD_PREFIX}{int(time.time())}.csv")
    _save_rows(rows, dest)
    new_dates = sorted({d for d, _, _ in rows})
    return {"saved": saved, "rejected": rejected, "filename": os.path.basename(dest), "rows": len(rows),
            "start_date": new_dates[0], "end_date": new_dates[-1], "excluded": excluded, "warnings": warnings,
            "unmatched": {"actual_only": ao, "forecast_only": fo}, "report": "\n".join(report)}


@router.get("/status")
def status():
    try:
        path = latest_upload()
    except FileNotFoundError:
        return {"exists": False}
    rows = load_rows(path)
    gaps = [r["GapMW"] for r in rows]
    mean = sum(gaps) / len(gaps)
    std = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5
    unmatched = None
    if os.path.exists("data/unmatched.json"):
        import json
        with open("data/unmatched.json", encoding="utf-8") as f:
            unmatched = json.load(f)
    return {"exists": True, "filename": os.path.basename(path), "rows": len(rows),
            "start_date": rows[0]["Date"], "end_date": rows[-1]["Date"], "unmatched": unmatched,
            "gap_mean_mw": round(mean, 1), "gap_std_mw": round(std, 1),
            "gap_min_mw": round(min(gaps), 1), "gap_max_mw": round(max(gaps), 1),
            "under_forecast_share": round(sum(g < 0 for g in gaps) / len(gaps), 3)}


@router.get("/history")
def history(start: str, end: str, lead: int = SEQ_LEN):
    try:
        path = latest_upload()
    except FileNotFoundError:
        raise HTTPException(404, "업로드된 데이터가 없습니다")
    try:
        s = (dt.date.fromisoformat(start) - dt.timedelta(days=lead)).isoformat()
        dt.date.fromisoformat(end)
    except ValueError:
        raise HTTPException(422, "start/end는 YYYY-MM-DD 형식이어야 합니다")
    rows = rows_between(load_rows(path), s, end)
    return {"history": [{"date": r["Date"], "gap_mw": r["GapMW"], "forecast_mw": r["ForecastMW"]} for r in rows],
            "n": len(rows), "lead": lead}
