"""
제주 저녁 갭 예측용 LSTM 아키텍처 (Day1 baseline·Day2 MLflow 학습·Day3 fine-tuning 공유).

명세서 v2 4장: 731일·약 700시퀀스에 1.6만 파라미터 3층 LSTM은 과하다. 갭의 자기상관은
lag1 0.55, lag2 0.27이고 그 뒤는 0에 가까워 예측 정보가 사실상 "전날 갭" 하나이므로
**소형 단층 LSTM을 기본**으로 두고, 실습의 3층 구성은 비교군(`arch="deep"`)으로만 남긴다.
"""
import os

from tensorflow import keras

from data.features import SEQ_LEN

N_FEATURES = 2  # (GapMW, ForecastMW)
DEFAULT_ARCH = os.getenv("LSTM_ARCH", "small")


def build_model(arch: str = DEFAULT_ARCH) -> keras.Model:
    if arch == "deep":
        layers = [
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(32, return_sequences=True),
            keras.layers.LSTM(16),
            keras.layers.Dense(16, activation="relu"),
        ]
    else:
        layers = [keras.layers.LSTM(16), keras.layers.Dense(8, activation="relu")]
    model = keras.Sequential([keras.layers.Input(shape=(SEQ_LEN, N_FEATURES)), *layers, keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model
