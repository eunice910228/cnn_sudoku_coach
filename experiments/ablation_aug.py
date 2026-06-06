# -*- coding: utf-8 -*-
"""
ablation_aug.py — 方向二:資料增強消融實驗(Ablation Study)
─────────────────────────────────────────────────────────────
digit_dataset.py 對合成數字加了平移/旋轉/模糊/雜訊擾動。但「加了增強」
到底有沒有用?消融實驗就是把某個設計拿掉、其餘不變,用數字證明它的價值。

本腳本訓練兩個結構完全相同的 CNN:
  A) 有增強(augment=True)    B) 無增強(augment=False,乾淨置中數字)
再比較兩者在「真實照片 + 逐級劣化」下的準確率——驗證增強帶來的穩健性。

執行:python experiments/ablation_aug.py
產出:figures/ablation_bar.png、figures/ablation_robustness.png
"""
import numpy as np

import _common as C
from robustness import deg_blur, deg_noise, deg_light, accuracy


def train_one(augment, tag):
    from vision.digit_dataset import make_dataset
    from vision.digit_cnn import build_model
    print(f"\n▶ 訓練模型【{tag}】(augment={augment})...")
    X, y = make_dataset(per_class=800, seed=42, augment=augment)
    n = int(len(X) * 0.9)
    model = build_model()
    model.fit(X[:n], y[:n], epochs=4, batch_size=128,
              validation_split=0.1, verbose=2)
    # 兩者都用「相同的、帶擾動的」測試集評估,才公平
    Xt, yt = make_dataset(per_class=200, seed=999, augment=True)
    acc = (np.argmax(model.predict(Xt, verbose=0), axis=1) == yt).mean()
    print(f"  {tag} 在(帶擾動)合成測試集準確率:{acc:.2%}")
    return model, acc


def main():
    out = C.ensure_out()
    C.use_cjk_font()
    import matplotlib.pyplot as plt

    _, board = C.load_real_cells("01.jpg")
    truth = C.load_labels("01.txt")

    m_aug, synth_aug = train_one(True, "有增強")
    m_clean, synth_clean = train_one(False, "無增強")

    # 真實照片(乾淨)準確率
    real_aug = accuracy(m_aug, board, truth)
    real_clean = accuracy(m_clean, board, truth)

    print("\n" + "=" * 56)
    print("  消融結果總表")
    print("=" * 56)
    print(f"  {'':12}{'有增強':>10}{'無增強':>10}")
    print(f"  {'合成測試集':10}{synth_aug:>10.2%}{synth_clean:>10.2%}")
    print(f"  {'真實照片':12}{real_aug:>10.2%}{real_clean:>10.2%}")
    print("=" * 56)

    # ---- 圖 A:長條對照 ----
    fig, ax = plt.subplots(figsize=(6, 4.5))
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w/2, [synth_aug, real_aug], w, label="有增強", color="#4C72B0")
    ax.bar(x + w/2, [synth_clean, real_clean], w, label="無增強",
           color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(["合成測試集\n(帶擾動)", "真實照片\n01.jpg"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("準確率")
    ax.set_title("資料增強消融:有 vs 無增強")
    for i, (a, b) in enumerate([(synth_aug, synth_clean),
                                (real_aug, real_clean)]):
        ax.text(i - w/2, a + 0.01, f"{a:.0%}", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.01, f"{b:.0%}", ha="center", fontsize=9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{out}/ablation_bar.png", dpi=150)
    print(f"已輸出:{out}/ablation_bar.png")

    # ---- 圖 B:劣化穩健性對照(增強的真正價值在這裡) ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    tests = [("對焦模糊", deg_blur, [0, 2, 4, 6, 8]),
             ("感光雜訊", deg_noise, [0, 20, 40, 60, 80]),
             ("光線不均", deg_light, [0, .3, .5, .7, .9])]
    for ax, (name, fn, levels) in zip(axes, tests):
        a_aug = [accuracy(m_aug, fn(board, s), truth) for s in levels]
        a_cln = [accuracy(m_clean, fn(board, s), truth) for s in levels]
        ax.plot(range(len(levels)), a_aug, "o-", color="#4C72B0",
                lw=2, label="有增強")
        ax.plot(range(len(levels)), a_cln, "s--", color="#C44E52",
                lw=2, label="無增強")
        ax.set_xticks(range(len(levels)))
        ax.set_xticklabels([str(s) for s in levels])
        ax.set_ylim(0, 1.05)
        ax.set_title(name)
        ax.set_ylabel("準確率")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("增強的價值:劣化條件下的穩健性對照", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/ablation_robustness.png", dpi=150)
    print(f"已輸出:{out}/ablation_robustness.png")


if __name__ == "__main__":
    main()
