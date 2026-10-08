"""
변동 크기별 보고서: naive(KPX 예측 그대로 = 갭 0) vs LSTM.

walk-forward 평가 결과(docs/eval_*.json)를 읽어 실제 갭 크기 구간별로
  - 방향 적중률(과소/과대): LSTM, persistence(어제 갭 방향 유지)
  - 평균 |갭|: 보정 전(= KPX 예측 그대로일 때 남는 평균 오차) vs LSTM 보정 후 남는 평균 |잔차| → 감소량(MW, %)
를 표로 만든다. 모델 구조나 성능 개선이 아니라 "보정이 얼마나 갭을 줄였는가"만 보고한다.

  python scripts/report_magnitude.py docs/eval_2025-07-01_2025-12-31.json [--out docs/report_by_magnitude.md]
"""
import argparse
import json
import math
import os
import sys

BUCKETS = [("\\|갭\\| < 40", lambda g: g < 40), ("40 ≤ \\|갭\\| < 60", lambda g: 40 <= g < 60),
           ("60 ≤ \\|갭\\| < 100", lambda g: 60 <= g < 100), ("\\|갭\\| ≥ 100", lambda g: g >= 100),
           ("\\|갭\\| ≥ 40 (누적)", lambda g: g >= 40), ("\\|갭\\| ≥ 60 (누적)", lambda g: g >= 60),
           ("\\|갭\\| ≥ 100 (누적)", lambda g: g >= 100), ("전체", lambda g: True)]


def report(path: str) -> str:
    e = json.load(open(path, encoding="utf-8"))
    y, lstm, pers = e["actual"], e["predictions"]["lstm"], e["predictions"]["persistence"]
    lines = [f"# 변동 크기별 보정 효과 — naive(KPX 예측 그대로) vs LSTM", "",
             f"평가 구간 {e['start']} ~ {e['end']} (walk-forward, n={len(y)}). 갭 = KPX 예측 − 실적(MW), 음수 = 과소예측.", "",
             "| 실제 갭 크기 | 일수 | 방향 적중 LSTM | 방향 적중 persistence | 보정 전 평균 \\|갭\\| | LSTM 보정 후 평균 \\|잔차\\| | 감소 (MW) | 감소율 | RMSE naive → LSTM |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, cond in BUCKETS:
        idx = [i for i, a in enumerate(y) if cond(abs(a))]
        if not idx:
            lines.append(f"| {name} | 0 | – | – | – | – | – | – | – |"); continue
        n = len(idx)
        hit_l = sum((y[i] < 0) == (lstm[i] < 0) for i in idx) / n
        hit_p = sum((y[i] < 0) == (pers[i] < 0) for i in idx) / n
        before = sum(abs(y[i]) for i in idx) / n
        after = sum(abs(y[i] - lstm[i]) for i in idx) / n
        rmse_b = math.sqrt(sum(y[i] ** 2 for i in idx) / n)
        rmse_a = math.sqrt(sum((y[i] - lstm[i]) ** 2 for i in idx) / n)
        lines.append(f"| {name} | {n} | {hit_l:.0%} | {hit_p:.0%} | {before:.1f} | {after:.1f} | {before - after:+.1f} | "
                     f"{(before - after) / before:.0%} | {rmse_b:.1f} → {rmse_a:.1f} |")
    lines += ["", "- 보정 전 평균 |갭| = KPX 예측을 그대로 썼을 때 남는 평균 오차 (naive). 보정 후 = KPX 예측 + LSTM 예측 갭을 썼을 때 남는 평균 오차.",
              "- 방향 적중 = 과소(갭<0)/과대(갭≥0)를 맞힌 비율. naive(갭 0)는 방향을 내지 않으므로 어제 갭 방향을 유지하는 persistence를 비교 기준으로 둔다."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eval_json")
    ap.add_argument("--out", default="docs/report_by_magnitude.md")
    args = ap.parse_args()
    text = report(args.eval_json)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
