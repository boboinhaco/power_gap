"""
Day1: FastAPI 요청/응답 Pydantic 스키마 (명세서 v2 5장).

/predict 는 "정답이 확정된 연속 20일"의 (gap_mw, forecast_mw) 시퀀스와 목표일을 받는다.
검증: 날짜 오름차순·중복 없음·연속·목표일보다 과거·숫자 유한·forecast_mw > 0 → 위반 시 422.
gap_mw 는 음수와 0을 허용한다 (실습 스키마의 close>0 제약은 여기서 제거).
"""
import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from data.features import SEQ_LEN


class GapPoint(BaseModel):
    date: dt.date
    gap_mw: float = Field(..., allow_inf_nan=False, description="18~21시 평균 (KPX 예측 − 실적), MW. 음수 = 과소예측")
    forecast_mw: float = Field(..., gt=0, allow_inf_nan=False, description="18~21시 평균 KPX 하루전 예측, MW")


def _check_sequence(points: list[GapPoint], require_consecutive: bool = True) -> None:
    dates = [p.date for p in points]
    if any(b <= a for a, b in zip(dates, dates[1:])):
        raise ValueError("날짜는 오름차순이어야 하며 중복될 수 없습니다")
    if require_consecutive and any((b - a).days != 1 for a, b in zip(dates, dates[1:])):
        raise ValueError("날짜는 하루 간격으로 연속되어야 합니다")


class PredictRequest(BaseModel):
    target_date: dt.date
    sequence: list[GapPoint] = Field(..., min_length=SEQ_LEN, max_length=SEQ_LEN,
                                     description=f"오래된 날 → 최신 날 순서의 연속 {SEQ_LEN}일")

    @model_validator(mode="after")
    def _validate(self):
        _check_sequence(self.sequence)
        if self.sequence[-1].date >= self.target_date:
            raise ValueError("시퀀스의 모든 날짜는 target_date보다 과거여야 합니다")
        return self


class PredictResponse(BaseModel):
    status: Literal["ok", "held"]
    target_date: dt.date
    predicted_gap_mw: float | None = None
    direction: Literal["under", "over", "near_zero"] | None = None
    model_version: str | None = None
    issued_at: str
    reason: str | None = None


class ReplayRequest(BaseModel):
    """과거 이력을 날짜순으로 재생한다. history[i] 예측에 history[i-20..i-1]을 쓰고 history[i]를 정답으로 기록."""
    history: list[GapPoint] = Field(..., min_length=SEQ_LEN + 1)
    label: Literal["real", "synthetic"] = "real"

    @field_validator("history")
    @classmethod
    def _consecutive(cls, v):
        _check_sequence(v)
        return v


class ReplayResponse(BaseModel):
    label: str
    n_predictions: int
    predictions: list[dict]
    events: list[dict]
    retrains: list[dict]
    drift_check: dict
