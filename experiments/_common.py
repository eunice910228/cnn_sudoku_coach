# -*- coding: utf-8 -*-
"""
_common.py — 實驗腳本共用工具
集中處理:專案路徑、載入真實照片的 81 格 + 正確答案、輸出資料夾。
所有 experiments/ 下的腳本都從這裡取資料,確保口徑一致。
"""
import os
import sys

import cv2
import numpy as np

# Windows 終端機預設 cp950,直接 print 中文/emoji 會亂碼或當掉 → 強制 UTF-8
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vision.grid_extractor import find_board, slice_cells   # noqa: E402

OUT_DIR = os.path.join(ROOT, "experiments", "figures")
MODEL_PATH = os.path.join(ROOT, "vision", "digit_cnn.h5")


def ensure_out():
    os.makedirs(OUT_DIR, exist_ok=True)
    return OUT_DIR


def load_real_cells(img_name="01.jpg"):
    """讀真實照片 → 切 81 格 → 回傳 (cells[81,28,28,1], board_gray[450,450])"""
    path = os.path.join(ROOT, img_name)
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"讀不到圖片:{path}")
    board = find_board(image)
    if board is None:
        raise RuntimeError(f"在 {img_name} 找不到盤面")
    return slice_cells(board), board


def load_labels(name="01.txt"):
    """讀 ground truth 盤面,回傳 (81,) int 陣列(0=空格)"""
    path = os.path.join(ROOT, "labels", name)
    digits = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digits += [int(x) for x in line.split()]
    if len(digits) != 81:
        raise ValueError(f"{name} 應有 81 格,實際 {len(digits)}")
    return np.array(digits, dtype="int64")


def load_model():
    from tensorflow import keras
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"找不到模型 {MODEL_PATH}\n請先到 vision/ 執行 python train.py")
    return keras.models.load_model(MODEL_PATH)


def use_cjk_font():
    """讓 matplotlib 顯示中文標題。設一整串候選(含 Windows 與 Linux/Colab 字型),
    matplotlib 會自動挑第一個『系統有裝』的;都沒有才退回豆腐(不會壞)。"""
    import matplotlib
    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PMingLiU",  # Windows
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",     # Colab/Linux
        "WenQuanYi Zen Hei", "Taipei Sans TC Beta", "Droid Sans Fallback",
    ]
    existing = matplotlib.rcParams.get("font.sans-serif", [])
    matplotlib.rcParams["font.sans-serif"] = candidates + list(existing)
    matplotlib.rcParams["axes.unicode_minus"] = False
