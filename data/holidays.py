"""
한국 공휴일·특수일 달력 (명세서 v2 5장 드리프트 경보 억제용).

실측 결과 제주 저녁 갭의 최대 오차는 모두 연휴에 몰려 있다
(2024-09-14~19 추석+폭염 -138~-187MW, 2025-10-03 개천절 -150MW, 2025-01-27 설 연휴 -136MW).
롤링 RMSE 계산에서 "특수일"을 제외한 값과 포함한 값을 둘 다 로그에 남긴다.

특수일 정의:
  1) 공휴일 당일
  2) 공휴일을 포함하는 3일 이상 연속 비근무일(주말+공휴일) 구간과 그 앞뒤 1일

※ 2026년 항목은 발표 전 공식 공고로 재확인할 것.
"""
import datetime as dt

PUBLIC_HOLIDAYS: dict[int, list[str]] = {
    2024: ["01-01", "02-09", "02-10", "02-11", "02-12", "03-01", "04-10", "05-06", "05-15",
           "06-06", "08-15", "09-16", "09-17", "09-18", "10-01", "10-03", "10-09", "12-25"],
    2025: ["01-01", "01-27", "01-28", "01-29", "01-30", "03-01", "03-03", "05-05", "05-06",
           "06-03", "06-06", "08-15", "10-03", "10-05", "10-06", "10-07", "10-08", "10-09", "12-25"],
    2026: ["01-01", "02-16", "02-17", "02-18", "03-01", "03-02", "05-05", "05-24", "05-25",
           "06-03", "06-06", "08-15", "08-17", "09-24", "09-25", "09-26", "10-03", "10-05",
           "10-09", "12-25"],
}


def _d(x) -> dt.date:
    return x if isinstance(x, dt.date) else dt.date.fromisoformat(str(x)[:10])


_HOLIDAY_SET = {dt.date.fromisoformat(f"{y}-{md}") for y, l in PUBLIC_HOLIDAYS.items() for md in l}


def is_holiday(day) -> bool:
    return _d(day) in _HOLIDAY_SET


def is_non_working(day) -> bool:
    d = _d(day)
    return d.weekday() >= 5 or d in _HOLIDAY_SET


def special_days(start, end) -> set[dt.date]:
    """[start, end] 안의 특수일 집합."""
    s, e = _d(start), _d(end)
    out = set()
    # 구간 밖 연휴가 걸칠 수 있으므로 앞뒤 10일 여유를 두고 스캔
    day = s - dt.timedelta(days=10)
    last = e + dt.timedelta(days=10)
    run: list[dt.date] = []

    def flush(run):
        if len(run) >= 3 and any(d in _HOLIDAY_SET for d in run):
            for d in run:
                out.add(d)
            out.add(run[0] - dt.timedelta(days=1))
            out.add(run[-1] + dt.timedelta(days=1))

    while day <= last:
        if is_non_working(day):
            run.append(day)
        else:
            flush(run)
            run = []
        if day in _HOLIDAY_SET:
            out.add(day)
        day += dt.timedelta(days=1)
    flush(run)
    return {d for d in out if s <= d <= e}


def is_special(day) -> bool:
    d = _d(day)
    return d in special_days(d, d)
