"""
Day2: MLflow 학습 → 기록 → 게이트(검증 walk-forward RMSE < naive 최저) → 등록 → alias `champion`.
Day3: 드리프트 감지 후 champion 가중치에서 이어서 학습하는 fine-tuning 재학습.

명세서 v2 4장·5장
  - 학습 2024-01-01~12-31, 검증 2025-01-01~06-30 walk-forward (최종평가 하반기는 scripts/evaluate_walkforward.py로 1회)
  - 스테이지(Production) 대신 alias(`models:/JejuGapPredictor@champion`)로 승격·로드
  - fine_tune: 최근 RETRAIN_ROWS(60)행 = 학습 [D-60, D-15] + 검증 [D-14, D-1]. 후보와 champion을 같은 검증 14일에서 비교.
    업로드 이력이 짧으면 retrain_trigger.rows_for_retrain 이 기준 파일(2024~2025)에서 앞 구간을 붙인다

실행:
    python scripts/train_baseline_v1.py               # 최초 1회 (scaler.pkl, thresholds.json)
    python serving_app/train_and_register.py          # 게이트 통과 시 champion 승격
    GATE_OVERRIDE=1 python serving_app/train_and_register.py   # 게이트 실패해도 승격 (MLflow·응답에 override 표시)
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlflow
import mlflow.tensorflow
import numpy as np
from mlflow.tracking import MlflowClient
from tensorflow import keras

from data.features import (GapScaler, build_sequences, load_rows, rows_between,
                           TRAIN_START, TRAIN_END, VAL_START, VAL_END)
from data.metrics import mae, rmse
from data.storage import latest_upload
from serving_app.evaluation import (compute_thresholds, format_table, gate_check, lstm_predict_fn,
                                    save_thresholds, walkforward)
from serving_app.lstm_model import build_model

SEED = 42
keras.utils.set_random_seed(SEED)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = "JejuGapPredictor"
ALIAS = "champion"
SCALER_PATH = "serving_app/models/scaler.pkl"
BASE_EPOCHS = int(os.getenv("BASE_EPOCHS", "120"))  # 검증 walk-forward로 고른 값 (소형 LSTM, 시드 고정)
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-4
RETRAIN_ROWS = int(os.getenv("RETRAIN_ROWS", "60"))  # 재학습에 쓰는 최근 행 수 (학습 + 검증)
FT_VAL_DAYS = 14
# 재학습 게이트 최소 개선폭(%). 0 = 후보 RMSE가 champion보다 조금이라도 낮으면 승격(기본).
# 14일 검증 표본의 잡음을 걸러내려면 2~5 정도로 올린다. 예: GATE_MIN_GAIN_PCT=3
FT_MIN_GAIN_PCT = float(os.getenv("GATE_MIN_GAIN_PCT", "0"))
FT_TRAIN_DAYS = RETRAIN_ROWS  # [D-60, D-15] 학습(46일) / [D-14, D-1] 검증(14일)
FT_MIN_TRAIN_ROWS = RETRAIN_ROWS - FT_VAL_DAYS - 6  # 제외일 몇 개는 허용


def _promote(run_id: str, note: str) -> int:
    v = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, ALIAS, v.version)
    client.set_model_version_tag(MODEL_NAME, v.version, "gate", note)
    return int(v.version)


def train_and_register(csv_path: str | None = None) -> dict:
    rows = load_rows(csv_path or latest_upload())
    train_rows = rows_between(rows, TRAIN_START, TRAIN_END)
    scaler = GapScaler.load(SCALER_PATH)
    X, y, _, _ = build_sequences(rows, scaler, target_start=TRAIN_START, target_end=TRAIN_END)
    X_train = np.array(X, dtype="float32")
    y_train_scaled = np.array([scaler.scale_gap(v) for v in y], dtype="float32")

    override = os.getenv("GATE_OVERRIDE", "0") == "1"
    with mlflow.start_run(run_name="base-train") as run:
        model = build_model()
        model.fit(X_train, y_train_scaled, epochs=BASE_EPOCHS, batch_size=16, verbose=0)
        pf = lstm_predict_fn(model, scaler)
        val = walkforward(rows, scaler, train_rows, VAL_START, VAL_END, pf)
        gate = gate_check(val["metrics"])
        print(format_table(val["metrics"], f"[VAL walk-forward {VAL_START}~{VAL_END}]"))

        mlflow.log_params({"mode": "scratch", "epochs": BASE_EPOCHS, "arch": os.getenv("LSTM_ARCH", "small"),
                           "train": f"{TRAIN_START}~{TRAIN_END}", "val": f"{VAL_START}~{VAL_END}",
                           "gate_reference": gate["reference"], "gate_override": override})
        for k, m in val["metrics"].items():
            mlflow.log_metric(f"val_rmse_{k}", m["rmse"]); mlflow.log_metric(f"val_mae_{k}", m["mae"])
            mlflow.log_metric(f"val_dirhit_{k}", m["direction_hit"])
            mlflow.log_metric(f"val_wape_{k}", m["wape"]); mlflow.log_metric(f"val_bias_{k}", m["bias"])
        mlflow.log_metric("rmse", val["metrics"]["lstm"]["rmse"])
        mlflow.log_metric("gate_improvement_pct", gate["improvement_pct"])
        mlflow.tensorflow.log_model(model, name="model", input_example=X_train[:1])

        train_eval = walkforward(rows, scaler, train_rows, TRAIN_START, TRAIN_END, pf)
        th = compute_thresholds(train_rows, train_eval)
        save_thresholds(th)
        mlflow.log_artifact("serving_app/models/thresholds.json")

        result = {"run_id": run.info.run_id, "rmse": gate["candidate_rmse"], "reference_rmse": gate["reference_rmse"],
                  "gate_passed": gate["passed"], "promoted": False}
        if gate["passed"] or override:
            note = "passed" if gate["passed"] else "override"
            result["version"] = _promote(run.info.run_id, note)
            result["promoted"] = True
            print(f"[GATE {'PASSED' if gate['passed'] else 'OVERRIDE'}] rmse lstm={gate['candidate_rmse']:.1f} vs naive[{gate['reference']}]="
                  f"{gate['reference_rmse']:.1f} MW, mae {gate['candidate_mae']:.1f} vs naive[{gate['mae_reference']}]={gate['reference_mae']:.1f} MW "
                  f"-> {MODEL_NAME} v{result['version']} = {ALIAS}")
        else:
            print(f"[GATE FAILED] lstm={gate['candidate_rmse']:.1f} >= naive[{gate['reference']}]={gate['reference_rmse']:.1f} MW "
                  f"-> 승격 차단 (GATE_OVERRIDE=1 로 시연용 강제 승격 가능)")
        return result


def fine_tune(rows: list[dict], as_of: str) -> dict:
    """
    Day3: champion 가중치에서 warm start. as_of = 트리거 시점 날짜 D (문자열 YYYY-MM-DD).
    학습 [D-90, D-15], 검증 [D-14, D-1]. 후보 vs champion을 같은 검증 날짜에서 비교.
    """
    D = dt.date.fromisoformat(as_of)
    tr_s, tr_e = (D - dt.timedelta(days=FT_TRAIN_DAYS)).isoformat(), (D - dt.timedelta(days=FT_VAL_DAYS + 1)).isoformat()
    va_s, va_e = (D - dt.timedelta(days=FT_VAL_DAYS)).isoformat(), (D - dt.timedelta(days=1)).isoformat()
    scaler = GapScaler.load(SCALER_PATH)
    train_rows = rows_between(rows, tr_s, tr_e)
    result = {"as_of": as_of, "train": [tr_s, tr_e], "val": [va_s, va_e], "promoted": False}
    if len(train_rows) < FT_MIN_TRAIN_ROWS:
        result["status"] = f"held: 학습 구간 {len(train_rows)}행 < {FT_MIN_TRAIN_ROWS}"
        return result

    X, y, _, _ = build_sequences(rows, scaler, target_start=tr_s, target_end=tr_e)
    Xv, yv, dv, _ = build_sequences(rows, scaler, target_start=va_s, target_end=va_e)
    if not X or not Xv:
        result["status"] = "held: 시퀀스를 만들 수 없음 (제외일 또는 데이터 부족)"
        return result
    X_train = np.array(X, dtype="float32")
    y_train_scaled = np.array([scaler.scale_gap(v) for v in y], dtype="float32")

    champion = mlflow.tensorflow.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
    champ_pred = lstm_predict_fn(champion, scaler)(Xv)
    champ_rmse, champ_mae = rmse(yv, champ_pred), mae(yv, champ_pred)
    champion.compile(optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR), loss="mse")

    with mlflow.start_run(run_name="fine-tune") as run:
        champion.fit(X_train, y_train_scaled, epochs=FINE_TUNE_EPOCHS, batch_size=16, verbose=0)
        cand_rmse = rmse(yv, lstm_predict_fn(champion, scaler)(Xv))
        mlflow.log_params({"mode": "fine-tune", "epochs": FINE_TUNE_EPOCHS, "as_of": as_of,
                           "train": f"{tr_s}~{tr_e}", "val": f"{va_s}~{va_e}", "n_train_rows": len(train_rows)})
        gain_pct = (champ_rmse - cand_rmse) / champ_rmse * 100 if champ_rmse else 0.0
        mlflow.log_metric("rmse", cand_rmse); mlflow.log_metric("champion_rmse", champ_rmse)
        mlflow.log_metric("gain_pct", gain_pct); mlflow.log_param("min_gain_pct", FT_MIN_GAIN_PCT)
        mlflow.tensorflow.log_model(champion, name="model", input_example=X_train[:1])
        pass_rmse = cand_rmse < champ_rmse and gain_pct > FT_MIN_GAIN_PCT
        pass_mae = cand_mae < champ_mae
        result.update({"run_id": run.info.run_id, "rmse": cand_rmse, "champion_rmse": champ_rmse,
                       "mae": cand_mae, "champion_mae": champ_mae,
                       "gain_pct": gain_pct, "min_gain_pct": FT_MIN_GAIN_PCT,
                       "pass_rmse": pass_rmse, "pass_mae": pass_mae,
                       "gate_detail": f"rmse {'ok' if pass_rmse else 'x'} / mae {'ok' if pass_mae else 'x'}",
                       "n_val": len(yv), "n_train_rows": len(train_rows)})
        # 복합 게이트: 같은 검증 14일에서 RMSE(개선폭 > FT_MIN_GAIN_PCT)와 MAE가 **둘 다** champion보다 낮아야 승격.
        # RMSE 하나는 하루 큰 오차에 끌려갈 수 있으므로, 이상치에 둔감한 MAE로 한 번 더 확인한다.
        if pass_rmse and pass_mae:
            result["version"] = _promote(run.info.run_id,
                                         f"fine-tune rmse {cand_rmse:.1f}<{champ_rmse:.1f} ({gain_pct:+.1f}%), mae {cand_mae:.1f}<{champ_mae:.1f}")
            result["promoted"] = True
            result["status"] = "promoted"
        else:
            result["status"] = "gate_failed"
    return result


if __name__ == "__main__":
    train_and_register()
