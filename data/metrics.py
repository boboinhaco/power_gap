"""
평가 지표와 기준 모델 (명세서 v2 4장). 기준 모델 4개와 LSTM이 **같은 함수**를 거친다.

naive 기준 모델 (모두 1일 앞 예측, 학습 구간 통계만 사용):
  zero        : 갭 0MW  = KPX 공식 예측을 그대로 쓴다 (보정 없음)
  train_mean  : 학습 구간 평균 갭만큼 상수 보정
  persistence : 가장 최근 확정 갭을 그대로 다음 날 갭으로
"""
import math


def rmse(y_true, y_pred) -> float:
    n = len(y_true)
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(y_true, y_pred)) / n) if n else 0.0


def mae(y_true, y_pred) -> float:
    n = len(y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / n if n else 0.0


def wape(y_true, y_pred) -> float:
    """가중 절대 백분율 오차 = Σ|오차| / Σ|실제| (갭 단위에서는 참고 지표)."""
    denom = sum(abs(a) for a in y_true)
    return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / denom if denom else 0.0


def bias(y_true, y_pred) -> float:
    """평균 편향 = mean(예측 − 실제). 양수 = 갭을 실제보다 높게(과대 쪽), 음수 = 낮게(과소 쪽) 예측하는 치우침."""
    n = len(y_true)
    return sum(b - a for a, b in zip(y_true, y_pred)) / n if n else 0.0


def direction_hit(y_true, y_pred) -> float:
    n = len(y_true)
    return sum((a < 0) == (b < 0) for a, b in zip(y_true, y_pred)) / n if n else 0.0


def summarize(y_true, y_pred) -> dict:
    return {"rmse": rmse(y_true, y_pred), "mae": mae(y_true, y_pred),
            "wape": wape(y_true, y_pred), "bias": bias(y_true, y_pred),
            "direction_hit": direction_hit(y_true, y_pred), "n": len(y_true)}


def baseline_predictions(train_rows: list[dict], prev_gaps: list[float]) -> dict[str, list[float]]:
    """평가 표본별 기준 모델 예측값. prev_gaps[i] = 표본 i 목표일 전날의 확정 갭."""
    mean = sum(r["GapMW"] for r in train_rows) / len(train_rows)
    return {
        "zero": [0.0] * len(prev_gaps),
        "train_mean": [mean] * len(prev_gaps),
        "persistence": list(prev_gaps),
    }


NAIVE_MODELS = ("zero", "train_mean", "persistence")


def rolling_rmse(records: list[dict], window: int, exclude_dates: set | None = None) -> list[tuple[str, float, int]]:
    """
    날짜순 records([{date, predicted, actual}])에서 각 시점의 최근 window건 RMSE.
    exclude_dates에 든 날짜는 창에서 제외한다. 반환: [(date, rmse, n_used)]
    """
    out = []
    for i in range(len(records)):
        win = [r for r in records[max(0, i - window + 1):i + 1]
               if not exclude_dates or r["date"] not in exclude_dates]
        if not win:
            out.append((records[i]["date"], 0.0, 0)); continue
        out.append((records[i]["date"], rmse([r["actual"] for r in win], [r["predicted"] for r in win]), len(win)))
    return out
