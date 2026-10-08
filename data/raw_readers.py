"""
data/raw/ 원본(KPX 하루전 수요예측 XLSX, 전력거래소 시간별 실적 CSV)을 읽는 공용 리더.

scripts/build_dataset.py(시간별 long-format 병합)와 data/prepare_power_gap.py(학습·서빙용
일별 CSV)가 같은 리더를 쓴다. 표준 라이브러리만 사용한다 (openpyxl·pandas 불필요).

실제 원본에서 확인된 함정 (명세서 v2 3장):
  - XLSX: 1행 제목, 2행 헤더 `구분, 1h…24h`, 날짜는 `YYYYMMDD` 문자열(sharedStrings)
  - CSV 인코딩: 대부분 CP949, 일부 UTF-8 BOM → utf-8-sig 실패 시 cp949
  - 천 단위 쉼표 `"1,004.027"` (2025년 1~3분기 파일)
  - macOS 한글 파일명 NFD → NFC 정규화 후 비교 (glob에 NFC 리터럴을 쓰면 못 찾음)
"""
import csv
import io
import os
import re
import unicodedata
import zipfile
import xml.etree.ElementTree as ET

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HOURS = tuple(range(1, 25))


def nfc(name: str) -> str:
    return unicodedata.normalize("NFC", name)


def list_raw_files(raw_dir: str) -> list[tuple[str, str]]:
    """(NFC 정규화 파일명, 실제 경로) 목록. 수정 시각 순(오래된 → 최신)."""
    out = []
    for f in os.listdir(raw_dir):
        if f.startswith("."):
            continue
        out.append((nfc(f), os.path.join(raw_dir, f)))
    return sorted(out, key=lambda x: (os.path.getmtime(x[1]), x[0]))  # 오래된 파일 → 최신 파일 (최신이 중복 날짜를 이김)


def to_float(v) -> float | None:
    v = (v or "").strip().replace(",", "").replace('"', "")
    return float(v) if v else None


def read_actual_csv(path: str) -> dict[str, list[float | None]]:
    """시간별 실적 CSV → {'YYYY-MM-DD': [h1..h24]}"""
    raw = open(path, "rb").read()
    try:
        txt = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        txt = raw.decode("cp949")
    out = {}
    for row in csv.reader(io.StringIO(txt)):
        if not row or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", row[0].strip()) or len(row) < 25:
            continue
        vals = [to_float(x) for x in row[1:25]]
        if all(v is None for v in vals):
            continue
        out[row[0].strip()] = vals
    return out


def read_forecast_xlsx(path: str) -> dict[str, list[float | None]]:
    """
    KPX 하루전 수요예측 XLSX → {'YYYY-MM-DD': [h1..h24]}

    두 가지 레이아웃을 지원한다.
      (A) 2024·2025 파일: 헤더 `구분, 1h … 24h`, 행 = 날짜 + 24개 값
      (B) 2026 파일: 헤더 `구분, 1h, '', '', '', 2h, …` 아래 `1구간~4구간`(15분) 96개 값 + 최대/최소/가중평균.
          같은 시간의 4개 구간을 평균해 시간값으로 만든다 (빈 헤더 셀은 앞 시간 라벨을 이어받음).
    """
    z = zipfile.ZipFile(path)
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        ss = ["".join(t.text or "" for t in si.iter("{%s}t" % NS["m"]))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS)]
    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    sheet = root.find("m:sheetData", NS)
    if sheet is None:
        return {}

    def col_index(ref: str) -> int:
        n = 0
        for ch in re.match(r"[A-Z]+", ref).group(0):
            n = n * 26 + ord(ch) - 64
        return n

    rows = []
    for r in sheet.findall("m:row", NS):
        cells = {}
        for c in r.findall("m:c", NS):
            v = c.find("m:v", NS)
            val = "" if v is None else v.text
            if c.get("t") == "s":
                val = ss[int(val)]
            cells[col_index(c.get("r"))] = (val or "").strip()
        rows.append(cells)

    # 헤더 행: '구분'이 있는 행. 열 → 시간 매핑 (빈 셀은 직전 시간 라벨 이어받기)
    col_hour: dict[int, int] = {}
    for cells in rows:
        if cells.get(1) == "구분":
            last = None
            for ci in sorted(cells):
                if ci == 1:
                    continue
                m = re.fullmatch(r"(\d{1,2})h", cells[ci])
                if m:
                    last = int(m.group(1))
                    col_hour[ci] = last
                elif cells[ci] == "" and last is not None:
                    col_hour[ci] = last
                else:
                    last = None  # 최대/최소/가중평균 등은 제외
            break
    out = {}
    for cells in rows:
        d = cells.get(1, "")
        if not re.fullmatch(r"\d{8}", d):
            continue
        key = f"{d[:4]}-{d[4:6]}-{d[6:]}"
        if col_hour:
            buckets: dict[int, list[float]] = {}
            for ci, h in col_hour.items():
                f = to_float(cells.get(ci, ""))
                if f is not None:
                    buckets.setdefault(h, []).append(f)
            vals = [sum(buckets[h]) / len(buckets[h]) if buckets.get(h) else None for h in HOURS]
        else:  # 헤더를 못 찾으면 (A) 레이아웃 가정
            vals = [to_float(cells.get(ci, "")) for ci in range(2, 26)]
        if all(v is None for v in vals):
            continue
        out[key] = vals
    return out
