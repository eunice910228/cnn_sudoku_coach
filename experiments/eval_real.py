# -*- coding: utf-8 -*-
"""
eval_real.py — 方向一:Sim-to-Real 域偏移(domain shift)評估
─────────────────────────────────────────────────────────────
模型是用「合成字型」訓練的,卻要辨識「真實照片」。本腳本量化兩者落差:
  1. 合成測試集準確率(模型熟悉的分布)
  2. 真實照片 81 格準確率(沒見過的分布)
並輸出失敗格放大圖 + 預測/正解對照圖。

執行:python experiments/eval_real.py
產出:figures/real_vs_synth.png、figures/failure_cells.png、figures/grid_compare.png
"""
import numpy as np

import _common as C


def main():
    out = C.ensure_out()
    C.use_cjk_font()
    import matplotlib.pyplot as plt

    model = C.load_model()

    # ---- (1) 合成測試集準確率 ----
    from vision.digit_dataset import make_dataset
    Xs, ys = make_dataset(per_class=200, seed=777, augment=True)
    ps = np.argmax(model.predict(Xs, verbose=0), axis=1)
    synth_acc = (ps == ys).mean()

    # ---- (2) 真實照片準確率 ----
    cells, _ = C.load_real_cells("01.jpg")
    truth = C.load_labels("01.txt")
    probs = model.predict(cells, verbose=0)
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)

    correct = pred == truth
    real_acc = correct.mean()
    nonempty = truth != 0
    real_acc_digits = correct[nonempty].mean()

    print("=" * 56)
    print("  Sim-to-Real 域偏移評估")
    print("=" * 56)
    print(f"  合成測試集準確率 (模型熟悉的分布) : {synth_acc:6.2%}")
    print(f"  真實照片 81 格準確率             : {real_acc:6.2%}")
    print(f"  真實照片『非空格』準確率          : {real_acc_digits:6.2%}")
    print(f"  域偏移落差 (synthetic - real)    : {synth_acc - real_acc:6.2%}")
    print("-" * 56)

    wrong = np.where(~correct)[0]
    if len(wrong):
        print(f"  辨識錯誤的格子({len(wrong)} 格):")
        for i in wrong:
            r, c = i // 9, i % 9
            print(f"    R{r+1}C{c+1}: 預測 {pred[i]} (信心 {conf[i]:.0%}) "
                  f"/ 正解 {truth[i]}")
    else:
        print("  真實照片 81 格全部辨識正確 🎉")
    print("=" * 56)

    # ---- 圖 A:合成 vs 真實 準確率長條圖 ----
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(["合成測試集", "真實照片\n(全部)", "真實照片\n(非空格)"],
                  [synth_acc, real_acc, real_acc_digits],
                  color=["#4C72B0", "#C44E52", "#DD8452"])
    for b, v in zip(bars, [synth_acc, real_acc, real_acc_digits]):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01,
                f"{v:.1%}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("單格準確率")
    ax.set_title("Sim-to-Real:合成資料 vs 真實照片")
    fig.tight_layout()
    fig.savefig(f"{out}/real_vs_synth.png", dpi=150)
    print(f"已輸出:{out}/real_vs_synth.png")

    # ---- 圖 B:失敗格放大圖 ----
    if len(wrong):
        n = len(wrong)
        cols = min(n, 6)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes:
            ax.axis("off")
        for ax, i in zip(axes, wrong):
            r, cc = i // 9, i % 9
            ax.imshow(cells[i].reshape(28, 28), cmap="gray")
            ax.set_title(f"R{r+1}C{cc+1}\n預測 {pred[i]} / 正解 {truth[i]}",
                         fontsize=9, color="#C44E52")
            ax.axis("off")
        fig.suptitle("辨識失敗的格子(CNN 看到的 28×28 影像)", fontsize=12)
        fig.tight_layout()
        fig.savefig(f"{out}/failure_cells.png", dpi=150)
        print(f"已輸出:{out}/failure_cells.png")

    # ---- 圖 C:整盤 預測 vs 正解 對照 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, data, title in [(axes[0], pred, "CNN 預測"),
                            (axes[1], truth, "正確答案")]:
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        for k in range(10):
            lw = 2.2 if k % 3 == 0 else 0.6
            ax.axhline(k, color="black", lw=lw)
            ax.axvline(k, color="black", lw=lw)
        for i in range(81):
            r, cc = i // 9, i % 9
            v = data[i]
            if v == 0:
                continue
            is_wrong = not correct[i]
            color = "#C44E52" if is_wrong else "#333333"
            weight = "bold" if is_wrong else "normal"
            ax.text(cc + 0.5, r + 0.5, str(v), ha="center", va="center",
                    fontsize=15, color=color, fontweight=weight)
        ax.set_title(title)
    fig.suptitle("整盤辨識對照(紅字 = 辨識錯誤)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/grid_compare.png", dpi=150)
    print(f"已輸出:{out}/grid_compare.png")


if __name__ == "__main__":
    main()
