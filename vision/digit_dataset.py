# -*- coding: utf-8 -*-
"""
digit_dataset.py — 合成訓練資料產生器
數獨上的數字是「印刷體」,所以不用 MNIST(手寫),
而是用多種字型把 1~9 渲染出來 + 隨機擾動(平移/旋轉/模糊/雜訊),
幾分鐘就能生成上萬張帶標籤的訓練影像。類別 0 = 空格。
依賴:Pillow、numpy(Colab 內建)
"""
import glob
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu/*.ttf",       # Colab / Ubuntu 內建
    "/usr/share/fonts/truetype/liberation/*.ttf",
    "C:/Windows/Fonts/arial*.ttf",                   # Windows 內建(本機)
    "C:/Windows/Fonts/times*.ttf",
    "C:/Windows/Fonts/cour*.ttf",
    "C:/Windows/Fonts/consol*.ttf",
    "C:/Windows/Fonts/segoeui*.ttf",
    "C:/Windows/Fonts/verdana*.ttf",
    "C:/Windows/Fonts/tahoma*.ttf",
]


def _load_fonts(size_range=(16, 24)):
    paths = []
    for pattern in FONT_DIRS:
        paths += glob.glob(pattern)
    if not paths:
        raise RuntimeError("找不到字型檔,請確認執行環境(建議 Colab)")
    fonts = []
    for p in paths:
        for s in range(*size_range):
            try:
                fonts.append(ImageFont.truetype(p, s))
            except OSError:
                pass
    return fonts


_USE_RESIDUE = True   # 消融開關:設 False 可訓練『無邊框殘留』對照組


def _add_grid_residue(arr, rng, prob=0.6):
    """模擬切格沒削乾淨的『格線/外框殘留』:在邊緣畫上亮線。

    這是修掉『空格被誤判成數字』的關鍵——真實照片(尤其角落格)的空白格
    常殘留外框粗黑線,反相後變成亮線。原本訓練資料的空格是純黑、模型沒看過
    這種邊緣亮線,就會硬湊成數字。

    ⚠ 只加在『空格類別(0)』:若連數字類別也加,模型會被邊緣亮線帶偏、
    反而讀不出中央那個(可能對比很淡的)數字。空格只加殘留,讓模型學會
    『中央有東西=數字;只有邊緣有線=空格』。
    """
    if rng.random() > prob:
        return arr
    a = arr.copy()
    for edge in range(4):                    # 上、下、左、右各有機會殘留
        if rng.random() < 0.4:
            t = rng.randint(2, 6)            # 線粗 2~5 px
            val = rng.uniform(0.6, 1.0)      # 亮度(對比深淺不一)
            if edge == 0:
                a[:t, :] = val
            elif edge == 1:
                a[-t:, :] = val
            elif edge == 2:
                a[:, :t] = val
            else:
                a[:, -t:] = val
    return a


def make_dataset(per_class=2000, seed=42, augment=True):
    """回傳 X: (N,28,28,1) float32、y: (N,) int — 類別 0=空格、1~9=數字

    augment=True 時加入隨機平移/旋轉/模糊/雜訊(模擬真實照片的擾動);
    augment=False 則渲染「乾淨置中」的數字——供消融實驗對照組使用。
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    fonts = _load_fonts()
    X, y = [], []

    for digit in range(10):
        for _ in range(per_class):
            img = Image.new("L", (28, 28), color=0)
            if digit != 0:
                draw = ImageDraw.Draw(img)
                font = rng.choice(fonts)
                # 置中(augment 時再加隨機平移)
                bbox = draw.textbbox((0, 0), str(digit), font=font)
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                jitter = rng.randint(-2, 2) if augment else 0
                x0 = (28 - w) // 2 - bbox[0] + jitter
                y0 = (28 - h) // 2 - bbox[1] + (rng.randint(-2, 2) if augment else 0)
                draw.text((x0, y0), str(digit), fill=255, font=font)
                if augment:
                    img = img.rotate(rng.uniform(-8, 8), fillcolor=0)
                    if rng.random() < 0.5:
                        img = img.filter(
                            ImageFilter.GaussianBlur(rng.uniform(0, 1)))
            arr = np.asarray(img, dtype="float32") / 255.0
            if augment:
                if digit == 0 and _USE_RESIDUE:                 # 只給空格加殘留
                    arr = _add_grid_residue(arr, np_rng)
                arr += np_rng.normal(0, 0.03, arr.shape)        # 感光雜訊
            X.append(np.clip(arr, 0, 1))
            y.append(digit)

    X = np.array(X, dtype="float32").reshape(-1, 28, 28, 1)
    y = np.array(y, dtype="int64")
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]
