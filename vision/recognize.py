# -*- coding: utf-8 -*-
"""
recognize.py — 照片 → 盤面(感知這一步,單獨抽出來)
─────────────────────────────────────────────────────────────
把『找盤面 → 切 81 格 → CNN 辨識』包成一個函式,讓桌面 App 和命令列工具共用。
需要 TensorFlow + 訓練好的模型(vision/digit_cnn.h5),建議在 Python 3.12 環境跑。
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "vision", "digit_cnn.h5")


def recognize_photo(img_path, model_path=MODEL_PATH):
    """回傳 (grid, conf):
        grid: 9x9 int(0=空格、1~9)
        conf: 9x9 float,每格的 softmax 信心(用來標『可疑、要人工確認』的格子)
    失敗時丟出例外(找不到圖/盤面/模型),由呼叫端友善處理。
    """
    import cv2
    import numpy as np
    from vision.grid_extractor import find_board, slice_cells

    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"讀不到圖片:{img_path}")
    board = find_board(image)
    if board is None:
        raise RuntimeError("找不到數獨盤面(請拍正一點、光線均勻)")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"找不到模型 {model_path}\n請先到 vision/ 執行 python train.py 或 restore_best.py")

    from tensorflow import keras
    model = keras.models.load_model(model_path)
    cells = slice_cells(board)
    probs = model.predict(cells, verbose=0)
    digits = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    grid = [[int(digits[r * 9 + c]) for c in range(9)] for r in range(9)]
    confg = [[float(conf[r * 9 + c]) for c in range(9)] for r in range(9)]
    return grid, confg


def suspect_cells(grid, conf, suspect_conf=0.90, hard_floor=0.45):
    """回傳值得人工確認的格子 [(r,c),...]:
    不確定的『數字』(可能讀錯)、或任何極低信心的格子。
    讀成空格、信心中等不算(空格本來就要自己填,殘留增強會讓空格信心偏低是正常的)。
    """
    out = []
    for r in range(9):
        for c in range(9):
            v, cf = grid[r][c], conf[r][c]
            if (v != 0 and cf < suspect_conf) or cf < hard_floor:
                out.append((r, c))
    return out


def grid_to_text(grid):
    """把 9x9 盤面轉成 puzzle.txt 風格的字串(空格=.)"""
    out = []
    for r in range(9):
        if r and r % 3 == 0:
            out.append("------+-------+------")
        row = []
        for c in range(9):
            if c and c % 3 == 0:
                row.append("|")
            row.append(str(grid[r][c]) if grid[r][c] else ".")
        out.append(" ".join(row))
    return "\n".join(out)
