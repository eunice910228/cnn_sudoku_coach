# -*- coding: utf-8 -*-
"""
extras.py — 其他加碼:特徵分群(t-SNE)+ 信心校準 × 矛盾偵測
─────────────────────────────────────────────────────────────
  1. t-SNE:把倒數第二層(128 維)特徵壓到 2D,看 1~9 是否自己分成 10 群——
     證明 CNN 學到的是「可分離的語意表徵」,不只是死背。
  2. 信心 × 白盒把關:展示 CNN 的 softmax 信心分布,並說明推理引擎如何
     用「矛盾偵測」替低信心/錯誤的辨識把關(神經符號架構的免費安全網)。

執行:python experiments/extras.py
產出:figures/tsne.png、figures/confidence.png
"""
import numpy as np

import _common as C


def main():
    out = C.ensure_out()
    C.use_cjk_font()
    import matplotlib.pyplot as plt
    import tensorflow as tf

    model = C.load_model()

    # ---- (1) t-SNE of penultimate (128-d) features ----
    from vision.digit_dataset import make_dataset
    X, y = make_dataset(per_class=150, seed=2024, augment=True)

    # 倒數第二層 = 最後一個 Dense(128) 之後、輸出層之前
    dense_layers = [l for l in model.layers if "dense" in l.name.lower()]
    penult = dense_layers[-2]                       # 128 維那層
    feat_model = tf.keras.Model(model.inputs, penult.output)
    feats = feat_model.predict(X, verbose=0)         # (N,128)
    print(f"倒數第二層特徵維度:{feats.shape}")

    from sklearn.manifold import TSNE
    print("跑 t-SNE 降維中(約十幾秒)...")
    emb = TSNE(n_components=2, init="pca", perplexity=30,
               random_state=0).fit_transform(feats)

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.get_cmap("tab10")
    for d in range(10):
        m = y == d
        label = "空格" if d == 0 else str(d)
        ax.scatter(emb[m, 0], emb[m, 1], s=8, color=cmap(d),
                   label=label, alpha=0.6)
    ax.legend(title="類別", markerscale=2, ncol=2, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("t-SNE:倒數第二層特徵(1~9 與空格自動分成 10 群)")
    fig.tight_layout()
    fig.savefig(f"{out}/tsne.png", dpi=150)
    print(f"已輸出:{out}/tsne.png")

    # ---- (2) 信心分布 + 白盒把關示意 ----
    cells, _ = C.load_real_cells("01.jpg")
    truth = C.load_labels("01.txt")
    probs = model.predict(cells, verbose=0)
    pred = np.argmax(probs, axis=1)
    conf = probs.max(axis=1)
    correct = pred == truth

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # 左:信心直方圖
    axes[0].hist(conf[correct], bins=20, range=(0, 1), color="#4C72B0",
                 alpha=0.8, label="辨識正確")
    if (~correct).any():
        axes[0].hist(conf[~correct], bins=20, range=(0, 1), color="#C44E52",
                     alpha=0.8, label="辨識錯誤")
    axes[0].set_xlabel("softmax 最高信心")
    axes[0].set_ylabel("格子數")
    axes[0].set_title(f"81 格信心分布(平均 {conf.mean():.1%})")
    axes[0].legend()

    # 右:模擬「CNN 認錯一格 → 推理引擎發現矛盾」
    # 把某一格故意改錯(複製一個同列已存在的數字),示意白盒如何抓到
    grid = [[int(pred[r * 9 + c]) for c in range(9)] for r in range(9)]
    msg = _contradiction_demo(grid)
    axes[1].axis("off")
    axes[1].text(0.02, 0.95, "白盒替黑盒把關(矛盾偵測)", fontsize=13,
                 fontweight="bold", va="top", transform=axes[1].transAxes)
    axes[1].text(0.02, 0.80, msg, fontsize=10, va="top", family="monospace",
                 transform=axes[1].transAxes)
    fig.tight_layout()
    fig.savefig(f"{out}/confidence.png", dpi=150)
    print(f"已輸出:{out}/confidence.png")


def _contradiction_demo(grid):
    """示範:把一格塞成與同列衝突的數字,驗證推理引擎能偵測矛盾"""
    import copy
    from reasoning.board import Board

    # 找第一個空格,塞入同列已出現的數字 → 製造矛盾
    bad = copy.deepcopy(grid)
    note = "(找不到可製造矛盾的格子)"
    for r in range(9):
        present = [v for v in grid[r] if v != 0]
        for c in range(9):
            if grid[r][c] == 0 and present:
                bad[r][c] = present[0]
                note = (f"模擬 CNN 把 R{r+1}C{c+1} 誤判為 {present[0]}\n"
                        f"(該列已存在 {present[0]})")
                break
        else:
            continue
        break

    b = Board(bad)
    ok = b.is_valid() if hasattr(b, "is_valid") else _basic_valid(bad)
    verdict = "❌ 偵測到矛盾 → 回報輸入有誤" if not ok else "✓ 未偵測到矛盾"
    return (f"{note}\n\n推理引擎檢查結果:\n  {verdict}\n\n"
            "結論:感知(CNN)會犯錯,\n但推理(符號)抓得到——\n"
            "這是神經符號架構免費送的安全網。")


def _basic_valid(grid):
    """後備:基本數獨合法性檢查(列/行/宮不得重複)"""
    def dup(vals):
        v = [x for x in vals if x != 0]
        return len(v) != len(set(v))
    for i in range(9):
        if dup(grid[i]) or dup([grid[r][i] for r in range(9)]):
            return False
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [grid[br + dr][bc + dc] for dr in range(3) for dc in range(3)]
            if dup(box):
                return False
    return True


if __name__ == "__main__":
    main()
