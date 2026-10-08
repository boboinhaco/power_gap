"""
walk-forward 평가 스크립트 (명세서 v2 4장). 기준 모델 4개 + LSTM이 같은 함수를 거친다.

  python scripts/evaluate_walkforward.py                 # 검증 구간(2025 상반기)
  python scripts/evaluate_walkforward.py --final         # 최종평가 구간(2025 하반기) — 모델·게이트 확정 후 1회만
  python scripts/evaluate_walkforward.py --model mlflow  # champion 모델로 평가 (기본: 로컬 jeju_gap_v1.keras)
  python scripts/evaluate_walkforward.py --start 2025-09-01 --end 2025-10-31
결과는 표로 출력하고 docs/eval_<start>_<end>.json 에 저장한다.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import (GapScaler, load_rows, rows_between, TRAIN_START, TRAIN_END,
                           VAL_START, VAL_END, TEST_START, TEST_END)
from data.storage import latest_upload
from serving_app.evaluation import format_table, gate_check, lstm_predict_fn, walkforward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--start"); ap.add_argument("--end")
    ap.add_argument("--model", choices=["local", "mlflow", "none"], default="local")
    ap.add_argument("--csv")
    args = ap.parse_args()
    start, end = (TEST_START, TEST_END) if args.final else (VAL_START, VAL_END)
    start, end = args.start or start, args.end or end

    rows = load_rows(args.csv or latest_upload())
    train_rows = rows_between(rows, TRAIN_START, TRAIN_END)
    scaler = GapScaler.load("serving_app/models/scaler.pkl")
    pf = None
    if args.model == "local":
        from tensorflow import keras
        pf = lstm_predict_fn(keras.models.load_model("serving_app/models/jeju_gap_v1.keras"), scaler)
    elif args.model == "mlflow":
        import mlflow, mlflow.tensorflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
        pf = lstm_predict_fn(mlflow.tensorflow.load_model("models:/JejuGapPredictor@champion"), scaler)

    ev = walkforward(rows, scaler, train_rows, start, end, pf)
    print(format_table(ev["metrics"], f"[walk-forward {start}~{end}] train={TRAIN_START}~{TRAIN_END}"))
    if pf:
        g = gate_check(ev["metrics"])
        print(f"gate(lstm < naive[{g['reference']}]): {g['candidate_rmse']:.1f} vs {g['reference_rmse']:.1f} MW "
              f"({g['improvement_pct']:+.1f}%) -> {'PASSED' if g['passed'] else 'FAILED'}")
    os.makedirs("docs", exist_ok=True)
    out = f"docs/eval_{start}_{end}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"start": start, "end": end, "model": args.model, "metrics": ev["metrics"],
                   "dates": ev["dates"], "actual": ev["actual"], "predictions": ev["predictions"]}, f, ensure_ascii=False, indent=1)
    print("saved ->", out)


if __name__ == "__main__":
    main()
