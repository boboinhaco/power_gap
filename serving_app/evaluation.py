"""
walk-forward 평가·게이트·임계값 산출 (명세서 v2 4장·5장). 학습 스크립트 3개가 공유한다.

  - walkforward(): naive 기준 모델 3개(zero/train_mean/persistence)와 LSTM이 **같은 표본·같은 함수**를 거친다
  - gate_check(): 검증 구간 walk-forward RMSE가 naive 3개 중 가장 낮은 값보다 낮으면 통과
  - compute_thresholds(): 방향 임계값(2024 갭 분포 33/67 분위)과 드리프트 임계값(2024 롤링 RMSE 95분위, 특수일 제외)
"""
import json
import os

import numpy as np

from data.features import build_sequences, rows_between
from data.holidays import special_days
from data.metrics import NAIVE_MODELS, baseline_predictions, rolling_rmse, summarize

THRESHOLDS_PATH = "serving_app/models/thresholds.json"
GATE_REFERENCE = "naive"  # zero/train_mean/persistence 중 RMSE 최저
DRIFT_WINDOW = 21


def lstm_predict_fn(model, scaler):
    def predict(X):
        if not X:
            return []
        out = model.predict(np.array(X, dtype="float32"), verbose=0).flatten()
        return [scaler.inverse_gap(float(p)) for p in out]
    return predict


def walkforward(rows, scaler, train_rows, start, end, predict_fn=None, extra: dict | None = None) -> dict:
    """start~end 목표일에 대한 1일 앞 예측 평가. 반환 metrics[모델명] = {rmse, mae, wape, direction_hit, n}"""
    X, y, dates, prev = build_sequences(rows, scaler, target_start=start, target_end=end)
    preds = baseline_predictions(train_rows, prev)
    if predict_fn is not None:
        preds["lstm"] = predict_fn(X)
    if extra:
        preds.update(extra)
    metrics = {k: summarize(y, v) for k, v in preds.items()}
    return {"start": start, "end": end, "dates": dates, "actual": y, "predictions": preds, "metrics": metrics}


def gate_check(metrics: dict, candidate: str = "lstm", reference: str = GATE_REFERENCE) -> dict:
    if reference == "naive":
        reference = min(NAIVE_MODELS, key=lambda k: metrics[k]["rmse"])
    c, r = metrics[candidate]["rmse"], metrics[reference]["rmse"]
    # 복합 게이트: RMSE뿐 아니라 MAE도 기준(naive 중 MAE 최저)보다 낮아야 통과. 한 지표만 좋아지는 우연을 거른다.
    mae_ref = min(NAIVE_MODELS, key=lambda k: metrics[k]["mae"])
    cm, rm = metrics[candidate]["mae"], metrics[mae_ref]["mae"]
    return {"passed": c < r and cm < rm, "passed_rmse": c < r, "passed_mae": cm < rm,
            "candidate": candidate, "candidate_rmse": c, "candidate_mae": cm,
            "reference": reference, "reference_rmse": r, "reference_mae": rm, "mae_reference": mae_ref,
            "improvement_pct": (r - c) / r * 100 if r else 0.0,
            "mae_improvement_pct": (rm - cm) / rm * 100 if rm else 0.0}


def format_table(metrics: dict, title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"  {'model':13s} {'RMSE':>7s} {'MAE':>7s} {'WAPE':>6s} {'dir':>5s} {'n':>4s}")
    for k, m in metrics.items():
        lines.append(f"  {k:13s} {m['rmse']:7.1f} {m['mae']:7.1f} {m['wape']:6.2f} {m['direction_hit']:5.2f} {m['n']:4d}")
    return "\n".join(lines)


def compute_thresholds(train_rows, train_eval: dict, model_key: str = "lstm", window: int = DRIFT_WINDOW) -> dict:
    gaps = sorted(r["GapMW"] for r in train_rows)
    q = lambda p: gaps[min(len(gaps) - 1, int(p * len(gaps)))]
    records = [{"date": d, "predicted": p, "actual": a}
               for d, p, a in zip(train_eval["dates"], train_eval["predictions"][model_key], train_eval["actual"])]
    excl = {d.isoformat() for d in special_days(train_eval["start"], train_eval["end"])}
    series = [r for _, r, n in rolling_rmse(records, window, excl) if n >= window // 2]
    series_all = [r for _, r, n in rolling_rmse(records, window) if n >= window // 2]
    s = sorted(series); sa = sorted(series_all)
    p95 = s[int(0.95 * (len(s) - 1))] if s else 0.0
    # 보조 판정 지표: 같은 창 규칙으로 계산한 롤링 MAE의 95분위 (RMSE 하나가 하루 큰 오차에 흔들리는 것을 거른다)
    from data.metrics import mae as _mae, rolling_metric
    sm = sorted(r for _, r, n in rolling_metric(records, window, _mae, excl) if n >= window // 2)
    mae_p95 = sm[int(0.95 * (len(sm) - 1))] if sm else 0.0
    return {
        "direction_under_mw": q(0.33), "direction_over_mw": q(0.67),
        "drift_window": window, "drift_rmse_threshold_mw": p95, "drift_mae_threshold_mw": mae_p95,
        "basis": {"period": [train_eval["start"], train_eval["end"]], "model": model_key,
                  "rolling_rmse_p50": s[len(s) // 2] if s else 0.0, "rolling_rmse_p95": p95,
                  "rolling_mae_p50": sm[len(sm) // 2] if sm else 0.0, "rolling_mae_p95": mae_p95,
                  "rolling_rmse_max": s[-1] if s else 0.0,
                  "rolling_rmse_p95_including_special": sa[int(0.95 * (len(sa) - 1))] if sa else 0.0,
                  "special_days_excluded": len(excl)},
    }


def save_thresholds(th: dict, path: str = THRESHOLDS_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(th, f, ensure_ascii=False, indent=2)


def load_thresholds(path: str = THRESHOLDS_PATH) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
