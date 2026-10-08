"""
제주 저녁 갭 데이터를 LSTM 입력 시퀀스로 변환하는 공용 유틸리티 (명세서 v2 4장).

정규화 CSV 1개(`Date,GapMW,ForecastMW`)만 읽는다.
  GapMW      : 목표일 18~21시 평균 (KPX 하루전 예측 − 실적), MW. 음수 = 과소예측
  ForecastMW : 같은 네 시간의 KPX 공식 예측 평균, MW

입력 시퀀스: 정답이 확정된 최근 SEQ_LEN(기본 3)일의 (GapMW, ForecastMW)
타깃: 그 다음 날의 GapMW

스케일러는 **학습 구간(2024년)에만 fit**한다. 전체 행으로 fit하면 미래 구간의
최대·최소를 학습 시점에 아는 데이터 누수가 된다 (실습 코드의 결함을 고친 지점).
"""
import csv
import datetime as dt
import math
import os
import pickle

SEQ_LEN = int(os.getenv("SEQ_LEN", "3"))  # 입력 시퀀스 길이 N. docs/seq_len_sweep.json: N=1~20 모두 ±1MW 안, N=3이 최종평가 최저(37.7MW)·시작 지연 최소
COLUMNS = ("Date", "GapMW", "ForecastMW")

# 명세서 v2 4장 분할
TRAIN_START, TRAIN_END = "2024-01-01", "2024-12-31"
VAL_START, VAL_END = "2025-01-01", "2025-06-30"
TEST_START, TEST_END = "2025-07-01", "2025-12-31"


def load_rows(csv_path: str) -> list[dict]:
    """CSV → 날짜순 [{Date, GapMW, ForecastMW}]. 컬럼·중복·비수치는 ValueError."""
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = [c.strip() for c in (reader.fieldnames or [])]
        missing = [c for c in COLUMNS if c not in fields]
        if missing:
            raise ValueError(f"CSV에 {missing} 컬럼이 없습니다 (필요: {list(COLUMNS)})")
        rows = []
        for r in reader:
            r = {k.strip(): v for k, v in r.items() if k}
            if not r.get("Date"):
                continue
            try:
                gap = float(r["GapMW"])
                fc = float(r["ForecastMW"])
            except (TypeError, ValueError):
                raise ValueError(f"{r.get('Date')}: GapMW/ForecastMW가 숫자가 아닙니다")
            if not (math.isfinite(gap) and math.isfinite(fc)) or fc <= 0:
                raise ValueError(f"{r.get('Date')}: 유한하지 않은 값이거나 ForecastMW <= 0")
            rows.append({"Date": r["Date"][:10], "GapMW": gap, "ForecastMW": fc})
    rows.sort(key=lambda r: r["Date"])
    dates = [r["Date"] for r in rows]
    if len(set(dates)) != len(dates):
        raise ValueError("중복 날짜가 있습니다")
    return rows


def rows_between(rows: list[dict], start: str | None, end: str | None) -> list[dict]:
    return [r for r in rows if (start is None or r["Date"] >= start) and (end is None or r["Date"] <= end)]


def is_consecutive(dates: list[str]) -> bool:
    ds = [dt.date.fromisoformat(d) for d in dates]
    return all((b - a).days == 1 for a, b in zip(ds, ds[1:]))


class GapScaler:
    """GapMW/ForecastMW를 각각 [0,1]로 정규화하는 min-max 스케일러 (갭은 음수 범위 포함)."""

    SCHEMA = "jeju-gap-v2"

    def __init__(self):
        self.gap_min = self.gap_max = None
        self.forecast_min = self.forecast_max = None
        self.fit_start = self.fit_end = None
        self.n_fit = 0

    def fit(self, rows: list[dict]) -> "GapScaler":
        gaps = [r["GapMW"] for r in rows]
        fcs = [r["ForecastMW"] for r in rows]
        self.gap_min, self.gap_max = min(gaps), max(gaps)
        self.forecast_min, self.forecast_max = min(fcs), max(fcs)
        self.fit_start, self.fit_end, self.n_fit = rows[0]["Date"], rows[-1]["Date"], len(rows)
        return self

    @staticmethod
    def _scale(value, lo, hi):
        return 0.0 if hi == lo else (value - lo) / (hi - lo)

    @staticmethod
    def _unscale(value, lo, hi):
        return value * (hi - lo) + lo

    def transform_point(self, gap_mw: float, forecast_mw: float) -> list[float]:
        return [self._scale(gap_mw, self.gap_min, self.gap_max),
                self._scale(forecast_mw, self.forecast_min, self.forecast_max)]

    def scale_gap(self, gap_mw: float) -> float:
        return self._scale(gap_mw, self.gap_min, self.gap_max)

    def inverse_gap(self, scaled: float) -> float:
        return self._unscale(scaled, self.gap_min, self.gap_max)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"schema": self.SCHEMA, **self.__dict__}, f)

    @classmethod
    def load(cls, path: str) -> "GapScaler":
        with open(path, "rb") as f:
            d = pickle.load(f)
        if d.pop("schema", None) != cls.SCHEMA:
            raise ValueError(f"{path}: 옛 스케일러 형식입니다. 삭제 후 train_baseline_v1.py로 다시 생성하세요")
        s = cls()
        s.__dict__.update(d)
        return s


def build_sequences(rows: list[dict], scaler: GapScaler, seq_len: int = SEQ_LEN,
                    target_start: str | None = None, target_end: str | None = None):
    """
    (seq_len, 2) 정규화 입력과 다음 날 GapMW(원단위) 타깃을 만든다.
    target_start/end를 주면 타깃 날짜가 그 범위인 표본만 만든다 (앞선 seq_len일은 범위 밖이어도 사용).

    반환: X (n, seq_len, 2), y (n,), target_dates (n,), prev_gaps (n,)  — prev_gaps는 persistence 기준용
    """
    scaled = [scaler.transform_point(r["GapMW"], r["ForecastMW"]) for r in rows]
    X, y, dates, prev = [], [], [], []
    for i in range(len(rows) - seq_len):
        t = rows[i + seq_len]["Date"]
        if (target_start and t < target_start) or (target_end and t > target_end):
            continue
        window_dates = [r["Date"] for r in rows[i:i + seq_len + 1]]
        if not is_consecutive(window_dates):
            continue  # 제외일이 창에 걸리면 표본에서 뺀다 (보간하지 않음)
        X.append(scaled[i:i + seq_len])
        y.append(rows[i + seq_len]["GapMW"])
        dates.append(t)
        prev.append(rows[i + seq_len - 1]["GapMW"])
    return X, y, dates, prev
