"""
[Day1] 로컬 baseline 모델 — scripts/train_baseline_v1.py

업로드된 전력 갭 CSV(Date,GapMW,ForecastMW)로
  1) 2024년 구간에만 스케일러 fit → serving_app/models/scaler.pkl
  2) 2024년 타깃으로 LSTM 학습 → serving_app/models/jeju_gap_v1.keras
  3) 2025 상반기 walk-forward 검증: naive 기준 3개 vs LSTM, 게이트(naive 최저 RMSE보다 낮을 것) 판정
  4) 방향·드리프트 임계값 → serving_app/models/thresholds.json

실행: (대시보드에서 data/jeju_gap.csv 업로드 후) python scripts/train_baseline_v1.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import (GapScaler, build_sequences, load_rows, rows_between,
                           TRAIN_START, TRAIN_END, VAL_START, VAL_END)
from data.storage import latest_upload
from serving_app.evaluation import (compute_thresholds, format_table, gate_check, lstm_predict_fn,
                                    save_thresholds, walkforward)
from serving_app.lstm_model import build_model

MODEL_PATH = "serving_app/models/jeju_gap_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = int(os.getenv("BASE_EPOCHS", "120"))  # 검증 walk-forward로 고른 값 (소형 LSTM, 시드 고정)
SEED = 42


def main():
    import numpy as np
    from tensorflow import keras
    keras.utils.set_random_seed(SEED)

    rows = load_rows(latest_upload())
    train_rows = rows_between(rows, TRAIN_START, TRAIN_END)
    if len(train_rows) < 60:
        sys.exit(f"학습 구간({TRAIN_START}~{TRAIN_END}) 행이 {len(train_rows)}개뿐입니다")

    scaler = GapScaler().fit(train_rows)
    scaler.save(SCALER_PATH)
    print(f"scaler fit on {len(train_rows)}행 ({TRAIN_START}~{TRAIN_END}) -> {SCALER_PATH}")

    X, y, dates, _ = build_sequences(rows, scaler, target_start=TRAIN_START, target_end=TRAIN_END)
    X_train = np.array(X, dtype="float32")
    y_train_scaled = np.array([scaler.scale_gap(v) for v in y], dtype="float32")
    print(f"train sequences: {len(X_train)} (targets {dates[0]} ~ {dates[-1]})")

    model = build_model()
    model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, batch_size=16, verbose=0)

    pf = lstm_predict_fn(model, scaler)
    val = walkforward(rows, scaler, train_rows, VAL_START, VAL_END, pf)
    print(format_table(val["metrics"], f"\n[VAL walk-forward {VAL_START}~{VAL_END}]"))
    gate = gate_check(val["metrics"])
    print(f"gate: lstm {gate['candidate_rmse']:.1f} vs naive[{gate['reference']}] {gate['reference_rmse']:.1f} MW "
          f"({gate['improvement_pct']:+.1f}%) -> {'PASSED' if gate['passed'] else 'FAILED'}")

    train_eval = walkforward(rows, scaler, train_rows, TRAIN_START, TRAIN_END, pf)
    th = compute_thresholds(train_rows, train_eval)
    save_thresholds(th)
    print(f"thresholds -> direction under<{th['direction_under_mw']:.1f} / over>{th['direction_over_mw']:.1f} MW, "
          f"drift rolling-{th['drift_window']} RMSE > {th['drift_rmse_threshold_mw']:.1f} MW (2024 p95, 특수일 제외)")

    model.save(MODEL_PATH)
    print(f"saved -> {MODEL_PATH}")
    if not gate["passed"]:
        print("※ 로컬 모델은 저장했지만 게이트 미통과. Day2 train_and_register.py에서 승격 여부를 정식 판정합니다.")


if __name__ == "__main__":
    main()
