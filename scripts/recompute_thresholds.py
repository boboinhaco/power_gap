"""
저장된 로컬 모델(serving_app/models/jeju_gap_v1.keras)로 thresholds.json만 다시 계산한다 (재학습 없음).
드리프트 판정 지표를 바꾸거나(RMSE+MAE) 창 길이를 바꿀 때 사용.
    SEQ_LEN=3 python scripts/recompute_thresholds.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

from data.features import GapScaler, TRAIN_END, TRAIN_START, load_rows, rows_between  # noqa: E402
from serving_app.evaluation import compute_thresholds, lstm_predict_fn, save_thresholds, walkforward  # noqa: E402

MODEL_PATH = "serving_app/models/jeju_gap_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"

if __name__ == "__main__":
    rows = load_rows("data/jeju_gap.csv")
    train_rows = rows_between(rows, TRAIN_START, TRAIN_END)
    scaler = GapScaler.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else GapScaler().fit(train_rows)
    model = tf.keras.models.load_model(MODEL_PATH)
    train_eval = walkforward(rows, scaler, train_rows, TRAIN_START, TRAIN_END, lstm_predict_fn(model, scaler))
    th = compute_thresholds(train_rows, train_eval)
    save_thresholds(th)
    print(f"drift rolling-{th['drift_window']}: RMSE > {th['drift_rmse_threshold_mw']:.1f} MW, "
          f"MAE > {th['drift_mae_threshold_mw']:.1f} MW (2024 p95, 특수일 {th['basis']['special_days_excluded']}일 제외)")
