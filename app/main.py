# -*- coding: utf-8 -*-
"""
main.py — 完整管線:照片 → 感知(CNN)→ 推理(邏輯引擎)→ 可解釋輸出
執行:python app/main.py 照片路徑.jpg
依賴:opencv-python、tensorflow(建議 Colab)
"""
import os
import sys

import cv2
import numpy as np
from tensorflow import keras

# Windows 終端機預設 cp950,直接 print 中文會亂碼 → 強制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 專案根目錄 = 本檔案(app/main.py)的上一層,不管從哪個資料夾執行都對得上
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from vision.grid_extractor import find_board, slice_cells          # noqa: E402
from reasoning.board import Board                                  # noqa: E402
from reasoning.solver import solve_with_explanations, rate_difficulty  # noqa: E402

MODEL_PATH = os.path.join(ROOT, "vision", "digit_cnn.h5")


def main(img_path):
    # ---------- 感知模組(神經:會看,黑盒) ----------
    image = cv2.imread(img_path)
    if image is None:
        sys.exit(f"讀不到圖片:{img_path}")
    board_img = find_board(image)
    if board_img is None:
        sys.exit("找不到數獨盤面(請拍正一點、光線均勻)")

    cells = slice_cells(board_img)
    model = keras.models.load_model(MODEL_PATH)
    probs = model.predict(cells, verbose=0)
    digits = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)

    grid = [[int(digits[r * 9 + c]) for c in range(9)] for r in range(9)]
    print("【CNN 辨識結果】(平均信心 {:.1%})".format(conf.mean()))
    print(Board(grid).pretty())

    # ---------- 推理模組(符號:會想,白盒) ----------
    solved, steps = solve_with_explanations(grid)
    print("\n【思路重播】")
    for i, s in enumerate(steps, 1):
        print(f"  {i:>2}. [{s.technique}] {s.reason}")

    print("\n【解答】")
    print(solved.pretty())
    label, used = rate_difficulty(steps)
    print(f"\n【難度評級】{label}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("用法:python app/main.py 照片路徑.jpg")
    main(sys.argv[1])
