"""
업로드된 전력 갭 데이터 파일 관리.

대시보드 또는 /data/upload로 올린 정규화 CSV(`Date,GapMW,ForecastMW`)는 data/uploads/에
`power_<timestamp>.csv`로 쌓인다. 학습·재학습·이력 재생은 항상 **접두사 `power_`가 붙은
가장 최근 파일**을 쓴다 (HAIC 실습의 `haic_*.csv`가 남아 있어도 선택되지 않도록 접두사로 거른다).
"""
import glob
import os

UPLOAD_DIR = "data/uploads"
UPLOAD_PREFIX = "power_"


def latest_upload(upload_dir: str = UPLOAD_DIR) -> str:
    files = sorted(glob.glob(os.path.join(upload_dir, f"{UPLOAD_PREFIX}*.csv")), key=os.path.getmtime)
    if not files:
        raise FileNotFoundError(
            "업로드된 전력 갭 데이터가 없습니다. 대시보드에서 data/jeju_gap.csv를 업로드하세요 "
            f"(python data/prepare_power_gap.py 로 생성 -> {upload_dir}/{UPLOAD_PREFIX}*.csv)."
        )
    return files[-1]
