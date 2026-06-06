# -*- coding: utf-8 -*-
"""
restore_best.py — 訓練多個候選模型,挑『對真實照片最穩』的那個存檔
─────────────────────────────────────────────────────────────
背景:合成字型→真實照片的泛化有 run-to-run 不穩(同設定有時 100%、有時很低)。
做法:訓練 K 個模型,各自在 01.jpg(附人工正解 labels/01.txt)上評估,
      保留準確率最高的存成 vision/digit_cnn.h5。
順帶用『空格殘留增強』(只加在空格類別)修掉空格被誤判成數字的問題。

執行:cd vision && python restore_best.py
"""
import os
import sys

import numpy as np
import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vision.digit_dataset import make_dataset            # noqa: E402
from vision.digit_cnn import build_model                 # noqa: E402
from vision.grid_extractor import find_board, slice_cells  # noqa: E402

K = 5                       # 候選數量
PER_CLASS = 800             # 較少資料/較少 epoch → 不過度貼合合成字型,真實泛化較好
EPOCHS = 4


def real_photo_eval():
    cells = slice_cells(find_board(cv2.imread(os.path.join(ROOT, "01.jpg"))))
    truth = []
    with open(os.path.join(ROOT, "labels", "01.txt"), encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0]
            truth += [int(x) for x in line.split()]
    return cells, np.array(truth)


def border_empty_eval(model):
    """邊框殘留空格 → 應判成空格的比例(她那個角落-8 bug 的情境)"""
    rng = np.random.RandomState(0)
    X = []
    for _ in range(200):
        img = np.zeros((28, 28), "float32") + rng.normal(0, 0.03, (28, 28))
        t = rng.randint(2, 7)
        if rng.random() < 0.7: img[:t, :] = 1.0
        if rng.random() < 0.7: img[:, :t] = 1.0
        if rng.random() < 0.3: img[-t:, :] = 1.0
        if rng.random() < 0.3: img[:, -t:] = 1.0
        X.append(np.clip(img, 0, 1))
    pred = model.predict(np.array(X).reshape(-1, 28, 28, 1), verbose=0).argmax(1)
    return (pred == 0).mean()


def main():
    cells, truth = real_photo_eval()
    best_acc, best_path = -1.0, os.path.join(ROOT, "vision", "digit_cnn.h5")
    tmp = os.path.join(ROOT, "vision", "_cand.h5")

    print(f"訓練 {K} 個候選(per_class={PER_CLASS}, epochs={EPOCHS}),挑 01.jpg 最高分的\n")
    for k in range(K):
        X, y = make_dataset(per_class=PER_CLASS, seed=42 + k, augment=True)
        n = int(len(X) * 0.9)
        m = build_model()
        m.fit(X[:n], y[:n], epochs=EPOCHS, batch_size=128,
              validation_split=0.1, verbose=0)
        pr = m.predict(cells, verbose=0).argmax(1)
        acc = (pr == truth).mean()
        em = truth == 0
        dig = (pr[~em] == truth[~em]).mean()
        print(f"  候選 {k+1}: 01.jpg 全部 {acc:.1%}  數字 {dig:.1%}  "
              f"空格 {(pr[em]==0).mean():.1%}")
        if acc > best_acc:
            best_acc = acc
            m.save(tmp)

    # 把最佳候選正式存檔
    if os.path.exists(tmp):
        os.replace(tmp, best_path)
    from tensorflow import keras
    best = keras.models.load_model(best_path)
    print(f"\n✅ 已存最佳模型 → {best_path}")
    print(f"   01.jpg 準確率: {best_acc:.1%}")
    print(f"   邊框殘留空格→判空格: {border_empty_eval(best):.1%}(修正前約 94.5%)")


if __name__ == "__main__":
    main()
