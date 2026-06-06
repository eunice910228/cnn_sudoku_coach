# -*- coding: utf-8 -*-
"""
make_report_figs.py — 生成 PPT 需要、但 experiments/ 還沒有的圖
  1. confusion_matrix.png  訓練後的混淆矩陣熱力圖(模型訓練)
  2. training_curve.png    訓練曲線:準確率/損失隨 epoch 變化(模型訓練)
  3. demo_mock.png         桌面 App 示意圖(demo):盤面 + 下一步提示高亮 + 思路面板
輸出到 experiments/figures/(跟其他圖放一起)。
執行:python report/make_report_figs.py
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import _common as C          # noqa: E402
OUT = C.ensure_out()


def cjk():
    C.use_cjk_font()


def make_training_and_confusion():
    import matplotlib.pyplot as plt
    from vision.digit_dataset import make_dataset
    from vision.digit_cnn import build_model

    print("訓練一個模型以取得『訓練曲線 + 混淆矩陣』...")
    X, y = make_dataset(per_class=600, seed=7, augment=True)
    n = int(len(X) * 0.9)
    model = build_model()
    hist = model.fit(X[:n], y[:n], epochs=6, batch_size=128,
                     validation_split=0.1, verbose=0)

    # 訓練曲線
    cjk()
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ep = range(1, len(hist.history["accuracy"]) + 1)
    ax[0].plot(ep, hist.history["accuracy"], "o-", label="訓練")
    ax[0].plot(ep, hist.history["val_accuracy"], "s--", label="驗證")
    ax[0].set_title("準確率隨 epoch 上升"); ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(ep, hist.history["loss"], "o-", label="訓練")
    ax[1].plot(ep, hist.history["val_loss"], "s--", label="驗證")
    ax[1].set_title("損失(loss)隨 epoch 下降"); ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss"); ax[1].legend(); ax[1].grid(alpha=.3)
    fig.suptitle("訓練曲線:模型在學習(損失下降、準確率上升)", fontsize=13)
    fig.tight_layout(); fig.savefig(f"{OUT}/training_curve.png", dpi=150)
    print(f"已輸出:{OUT}/training_curve.png")

    # 混淆矩陣
    Xt, yt = make_dataset(per_class=200, seed=321, augment=True)
    pred = model.predict(Xt, verbose=0).argmax(1)
    cm = np.zeros((10, 10), int)
    for t, p in zip(yt, pred):
        cm[t][p] += 1
    acc = (pred == yt).mean()
    cjk()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    labels = ["空"] + [str(i) for i in range(1, 10)]
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("預測"); ax.set_ylabel("真實")
    for i in range(10):
        for j in range(10):
            if cm[i][j]:
                ax.text(j, i, cm[i][j], ha="center", va="center",
                        color="white" if cm[i][j] > cm.max() * .5 else "black",
                        fontsize=8)
    ax.set_title(f"混淆矩陣(測試準確率 {acc:.1%}) 對角線=答對")
    fig.colorbar(im, fraction=.046)
    fig.tight_layout(); fig.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
    print(f"已輸出:{OUT}/confusion_matrix.png")


def make_demo_mock():
    """畫一張桌面 App 示意圖:盤面 + 下一步提示(黃)+ 思路面板"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    cjk()
    # 用內建範例題,示範『下一步提示在 R2C1』
    grid = [
        [9, 0, 0, 5, 0, 8, 0, 0, 7],
        [0, 8, 0, 3, 0, 2, 9, 0, 5],
        [0, 5, 4, 0, 0, 0, 0, 8, 0],
        [0, 7, 0, 6, 8, 0, 0, 3, 2],
        [1, 0, 0, 0, 0, 4, 0, 0, 8],
        [5, 0, 0, 2, 1, 9, 0, 6, 0],
        [0, 0, 0, 9, 0, 6, 0, 0, 1],
        [7, 2, 6, 0, 0, 1, 0, 4, 0],
        [0, 0, 1, 4, 7, 0, 0, 5, 6],
    ]
    hint = (1, 0)   # R2C1
    fig, (axb, axp) = plt.subplots(1, 2, figsize=(11, 5.6),
                                   gridspec_kw={"width_ratios": [1, .85]})
    axb.set_xlim(0, 9); axb.set_ylim(0, 9); axb.invert_yaxis()
    axb.set_xticks([]); axb.set_yticks([]); axb.set_aspect("equal")
    axb.add_patch(Rectangle((hint[1], hint[0]), 1, 1, color="#FFF1A8"))
    for k in range(10):
        lw = 2.4 if k % 3 == 0 else 0.6
        axb.axhline(k, color="black", lw=lw); axb.axvline(k, color="black", lw=lw)
    for r in range(9):
        for c in range(9):
            if grid[r][c]:
                axb.text(c + .5, r + .5, str(grid[r][c]), ha="center",
                         va="center", fontsize=16, color="#1a1a1a")
    axb.set_title("數獨教練 App(play_gui.py)", fontsize=13)

    axp.axis("off")
    panel = (
        "【下一步】看 R2C1(黃色格)\n\n"
        "招式:唯一候選數 (Naked Single)\n\n"
        "思路:R2C1 的候選數只剩 6\n"
        "(其餘 1~9 已出現在同列、\n"
        " 同行或同宮)→ 填入 6\n\n"
        "★ 第一次用到這招——心法:\n"
        "把這格同列、同行、同宮已出現的\n"
        "數字全排除後,只剩一個可能,\n"
        "那它就一定是它。\n\n"
        "(自己把它填上去,再按一次\n"
        " 要下一步)"
    )
    axp.text(0.0, 1.0, panel, va="top", ha="left", fontsize=11,
             bbox=dict(boxstyle="round", fc="#FFFFFF", ec="#90A4AE"))
    axp.text(0.0, -0.02, "互動模型:你填一格(任何順序)→ 讀『目前盤面』→ 只給下一步思路",
             va="top", ha="left", fontsize=9, color="#555")
    fig.suptitle("Live Demo:拍照→辨識→自己填→要提示就給下一步(不劇透)", fontsize=12)
    fig.tight_layout(); fig.savefig(f"{OUT}/demo_mock.png", dpi=150)
    print(f"已輸出:{OUT}/demo_mock.png")


if __name__ == "__main__":
    make_training_and_confusion()
    make_demo_mock()
