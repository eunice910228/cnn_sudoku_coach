# -*- coding: utf-8 -*-
"""
train.py — 訓練腳本(在 Colab 執行,CPU 數分鐘 / GPU 一分鐘內)
產出:digit_cnn.h5(模型),並於終端機印出測試準確率與混淆矩陣。
執行:python train.py
"""
import sys

import numpy as np

# Windows 終端機預設 cp950,印中文/符號會亂碼或當掉 → 強制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from digit_dataset import make_dataset
from digit_cnn import build_model


def main():
    print("▶ 生成合成資料集(印刷體數字 + 擾動)...")
    X, y = make_dataset(per_class=2000)
    n = int(len(X) * 0.9)
    X_train, y_train, X_test, y_test = X[:n], y[:n], X[n:], y[n:]
    print(f"  訓練 {len(X_train)} 張 / 測試 {len(X_test)} 張")

    model = build_model()
    model.summary()
    model.fit(X_train, y_train, epochs=5, batch_size=128,
              validation_split=0.1, verbose=2)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n▶ 測試集準確率:{acc:.4f}")

    # 混淆矩陣(列=真實標籤, 行=預測標籤)
    pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_test, pred):
        cm[t][p] += 1
    print("\n▶ 混淆矩陣(列=真實, 行=預測):")
    print("     " + " ".join(f"{i:>4}" for i in range(10)))
    for i, row in enumerate(cm):
        print(f"  {i}: " + " ".join(f"{v:>4}" for v in row))

    model.save("digit_cnn.h5")
    print("\n▶ 模型已存檔:digit_cnn.h5")


if __name__ == "__main__":
    main()
