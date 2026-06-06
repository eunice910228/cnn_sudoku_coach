# -*- coding: utf-8 -*-
"""
robustness.py — 方向一(進階):真實拍照條件壓力測試
─────────────────────────────────────────────────────────────
01.jpg 是乾淨的數位截圖,模型 100% 全對 → 看不到域偏移。
但真實「手機拍紙本」會有:對焦模糊、感光雜訊、光線不均、JPEG 壓縮。
本腳本對校正後的盤面施加「逐級加重」的這四種劣化,觀察 CNN 何時開始崩壞——
這才是 Sim-to-Real 落差真正現形的地方,也是 CNN 穩健性(robustness)的經典題材。

執行:python experiments/robustness.py
產出:figures/robustness_curves.png、figures/degradation_demo.png
"""
import cv2
import numpy as np

import _common as C


def deg_blur(board, s):
    k = int(s) * 2 + 1
    return cv2.GaussianBlur(board, (k, k), 0) if k > 1 else board


def deg_noise(board, s):
    noise = np.random.RandomState(0).normal(0, s, board.shape)
    return np.clip(board.astype("float32") + noise, 0, 255).astype("uint8")


def deg_light(board, s):
    """左暗右亮的線性光照梯度,s=明暗對比強度(0~1)"""
    h, w = board.shape
    grad = np.linspace(1 - s, 1 + s, w, dtype="float32")[None, :]
    return np.clip(board.astype("float32") * grad, 0, 255).astype("uint8")


def deg_jpeg(board, q):
    ok, enc = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def accuracy(model, board, truth):
    from vision.grid_extractor import slice_cells
    cells = slice_cells(board)
    pred = np.argmax(model.predict(cells, verbose=0), axis=1)
    return (pred == truth).mean()


# (名稱, 函式, 嚴重度序列, x 軸標籤)
CORRUPTIONS = [
    ("對焦模糊", deg_blur, [0, 1, 2, 3, 4, 5, 6], "高斯模糊核半徑"),
    ("感光雜訊", deg_noise, [0, 10, 20, 30, 40, 50, 70], "雜訊標準差"),
    ("光線不均", deg_light, [0, .2, .4, .6, .8, .9, .95], "明暗對比強度"),
    ("JPEG 壓縮", deg_jpeg, [100, 50, 30, 20, 12, 8, 5], "JPEG 品質(越低越糟)"),
]


def main():
    out = C.ensure_out()
    C.use_cjk_font()
    import matplotlib.pyplot as plt

    model = C.load_model()
    _, board = C.load_real_cells("01.jpg")
    truth = C.load_labels("01.txt")

    print("=" * 56)
    print("  真實拍照條件壓力測試(逐級加重)")
    print("=" * 56)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, fn, levels, xlabel) in zip(axes.ravel(), CORRUPTIONS):
        accs = [accuracy(model, fn(board, s), truth) for s in levels]
        print(f"\n  【{name}】")
        for s, a in zip(levels, accs):
            print(f"    {xlabel}={s:<6}: 準確率 {a:6.2%}")
        ax.plot(range(len(levels)), accs, "o-", color="#C44E52", lw=2)
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(s) for s in levels])
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="#4C72B0", ls="--", lw=1, label="乾淨基準 100%")
        ax.set_title(f"{name}:準確率隨劣化程度變化")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("81 格準確率")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
    fig.suptitle("CNN 穩健性:模型在哪種拍照劣化下開始崩壞", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/robustness_curves.png", dpi=150)
    print(f"\n已輸出:{out}/robustness_curves.png")

    # ---- 劣化示意圖:同一格在各種劣化最重時長什麼樣 ----
    fig, axes = plt.subplots(1, 5, figsize=(13, 3))
    axes[0].imshow(board, cmap="gray")
    axes[0].set_title("原始(乾淨)")
    demos = [("對焦模糊", deg_blur(board, 6)),
             ("感光雜訊", deg_noise(board, 70)),
             ("光線不均", deg_light(board, 0.95)),
             ("JPEG 壓縮", deg_jpeg(board, 5))]
    for ax, (name, im) in zip(axes[1:], demos):
        ax.imshow(im, cmap="gray")
        ax.set_title(name)
    for ax in axes:
        ax.axis("off")
    fig.suptitle("四種模擬拍照劣化(最重程度)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/degradation_demo.png", dpi=150)
    print(f"已輸出:{out}/degradation_demo.png")
    print("=" * 56)


if __name__ == "__main__":
    main()
