"""
모델 불러오기 — serving_app/model_loader.py (Day1 로컬 → Day2 MLflow alias)

환경변수
   LOADING_MODE = lazy(기본) | eager
   MODEL_SOURCE = local(기본, Day1) | mlflow(Day2~)   mlflow면 models:/JejuGapPredictor@champion
   MLFLOW_TRACKING_URI = sqlite:///mlflow.db (기본)

명세서 v2 5장: 스테이지 대신 alias, model_version은 레지스트리의 실제 버전 번호,
승격 후 invalidate_cache()로 다음 /predict가 새 버전을 쓰게 한다.
스케일러는 학습 때 2024년 구간으로 한 번 fit한 로컬 scaler.pkl을 항상 쓴다 (재fit 금지).
"""
import datetime as dt
import os
import time

from data.features import GapScaler

LOCAL_MODEL_PATH = "serving_app/models/jeju_gap_v1.keras"
SCALER_PATH = "serving_app/models/scaler.pkl"
MODEL_NAME = "JejuGapPredictor"
ALIAS = "champion"
MLFLOW_MODEL_URI = f"models:/{MODEL_NAME}@{ALIAS}"

_model_cache = None


class LoadedModel:
    def __init__(self, keras_model, scaler: GapScaler, version: str, source: str):
        self._keras_model = keras_model
        self.scaler = scaler
        self.version = version
        self.source = source
        self.loaded_at = dt.datetime.now().isoformat(timespec="seconds")

    def predict_one(self, sequence: list[dict]) -> float:
        """sequence = [{"gap_mw": -12.4, "forecast_mw": 795.0}, ... 20개] (오래된 날 → 최근 날) → 다음 날 GapMW"""
        import numpy as np
        scaled = [self.scaler.transform_point(p["gap_mw"], p["forecast_mw"]) for p in sequence]
        x = np.array([scaled], dtype="float32")
        pred_scaled = float(self._keras_model.predict(x, verbose=0)[0][0])
        return self.scaler.inverse_gap(pred_scaled)


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    return mlflow


def champion_version() -> str | None:
    """레지스트리에서 champion alias가 가리키는 버전 번호 (모델을 로드하지 않음)."""
    try:
        from mlflow.tracking import MlflowClient
        _mlflow()
        return str(MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS).version)
    except Exception:
        return None


def _load_from_local() -> LoadedModel:
    from tensorflow import keras
    return LoadedModel(keras.models.load_model(LOCAL_MODEL_PATH), GapScaler.load(SCALER_PATH), "v1-local", "local")


def _load_from_mlflow() -> LoadedModel:
    import mlflow.tensorflow
    _mlflow()
    keras_model = mlflow.tensorflow.load_model(MLFLOW_MODEL_URI)
    ver = champion_version()
    return LoadedModel(keras_model, GapScaler.load(SCALER_PATH), f"{MODEL_NAME} v{ver}" if ver else MODEL_NAME, "mlflow")


def _load_model() -> LoadedModel:
    return _load_from_mlflow() if os.getenv("MODEL_SOURCE", "local") == "mlflow" else _load_from_local()


def load_eager() -> LoadedModel:
    global _model_cache
    start = time.time()
    _model_cache = _load_model()
    print(f"[eager] model {_model_cache.version} loaded in {time.time() - start:.3f}s at startup")
    return _model_cache


def get_model() -> LoadedModel:
    global _model_cache
    if _model_cache is None:
        start = time.time()
        _model_cache = _load_model()
        print(f"[lazy] model {_model_cache.version} loaded in {time.time() - start:.3f}s on first request")
    return _model_cache


def invalidate_cache() -> None:
    """승격 직후 호출. 다음 get_model()이 새 champion을 다시 읽는다."""
    global _model_cache
    _model_cache = None


def is_loaded() -> bool:
    return _model_cache is not None


def current_version() -> str | None:
    return _model_cache.version if _model_cache else None
