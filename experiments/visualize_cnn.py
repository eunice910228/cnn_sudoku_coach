# -*- coding: utf-8 -*-
"""
visualize_cnn.py — 方向三:CNN 內部視覺化(教科書級素材)
─────────────────────────────────────────────────────────────
打開 CNN 的黑盒,看它「學到什麼」「看哪裡」:
  1. 第一層 32 個濾波器(filters)— 模型自學出的邊緣/筆畫偵測器
  2. 特徵圖(feature maps)— 真實數字餵進去後各通道的激活
  3. Grad-CAM 熱力圖 — 模型靠哪塊像素做出判斷

執行:python experiments/visualize_cnn.py
產出:figures/filters.png、figures/feature_maps.png、figures/gradcam.png
"""
import numpy as np

import _common as C


def pick_real_digit(cells, truth, want):
    """從真實照片挑一個指定數字的格子;沒有就回傳第一個非空格"""
    for i in range(81):
        if truth[i] == want:
            return cells[i]
    for i in range(81):
        if truth[i] != 0:
            return cells[i]
    return cells[0]


def main():
    out = C.ensure_out()
    C.use_cjk_font()
    import matplotlib.pyplot as plt
    import tensorflow as tf

    model = C.load_model()
    cells, _ = C.load_real_cells("01.jpg")
    truth = C.load_labels("01.txt")

    conv_layers = [l for l in model.layers if "conv" in l.name.lower()]
    print(f"卷積層:{[l.name for l in conv_layers]}")

    # ---- (1) 第一層濾波器 ----
    w = conv_layers[0].get_weights()[0]          # (3,3,1,32)
    n = w.shape[-1]
    fig, axes = plt.subplots(4, 8, figsize=(10, 5))
    for i, ax in enumerate(axes.ravel()):
        if i < n:
            f = w[:, :, 0, i]
            f = (f - f.min()) / (np.ptp(f) + 1e-8)
            ax.imshow(f, cmap="gray")
            ax.set_title(f"#{i}", fontsize=7)
        ax.axis("off")
    fig.suptitle("第一層 32 個濾波器(模型自學出的邊緣/筆畫偵測器)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/filters.png", dpi=150)
    print(f"已輸出:{out}/filters.png")

    # ---- (2) 特徵圖 ----
    digit_img = pick_real_digit(cells, truth, want=8)
    x = digit_img.reshape(1, 28, 28, 1)
    feat_model = tf.keras.Model(model.inputs, conv_layers[0].output)
    fmaps = feat_model.predict(x, verbose=0)[0]   # (26,26,32)
    fig, axes = plt.subplots(4, 9, figsize=(12, 5.5))
    axes[0, 0].imshow(digit_img.reshape(28, 28), cmap="gray")
    axes[0, 0].set_title("輸入數字", fontsize=9, color="#4C72B0")
    axes[0, 0].axis("off")
    flat = axes.ravel()
    for k in range(min(32, fmaps.shape[-1])):
        ax = flat[k + 1]
        ax.imshow(fmaps[:, :, k], cmap="viridis")
        ax.set_title(f"ch{k}", fontsize=7)
        ax.axis("off")
    for ax in flat:
        ax.axis("off")
    fig.suptitle("第一層特徵圖:同一個數字觸發了哪些濾波器", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/feature_maps.png", dpi=150)
    print(f"已輸出:{out}/feature_maps.png")

    # ---- (3) Grad-CAM ----
    # 在最後一層卷積處把模型切兩半:base 算到卷積輸出,head 接著算到分類。
    # 這樣才能在 tape 裡先 watch 卷積輸出,Keras 3 才追得到梯度。
    last_conv = conv_layers[-1]
    conv_idx = model.layers.index(last_conv)
    base = tf.keras.Model(model.inputs, last_conv.output)
    head_layers = model.layers[conv_idx + 1:]

    def head(t):
        for layer in head_layers:
            t = layer(t, training=False)
        return t

    samples = []
    for want in (1, 5, 8, 9):
        samples.append((want, pick_real_digit(cells, truth, want)))

    fig, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 6))
    for j, (want, img) in enumerate(samples):
        x = tf.convert_to_tensor(img.reshape(1, 28, 28, 1), dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_out = base(x, training=False)
            tape.watch(conv_out)
            preds = head(conv_out)
            cls = tf.argmax(preds[0])
            score = preds[:, cls]
        grads = tape.gradient(score, conv_out)[0]            # (H,W,C)
        weights = tf.reduce_mean(grads, axis=(0, 1))         # (C,)
        cam = tf.reduce_sum(conv_out[0] * weights, axis=-1)  # (H,W)
        cam = tf.nn.relu(cam).numpy()
        cam = (cam - cam.min()) / (np.ptp(cam) + 1e-8)
        import cv2
        cam = cv2.resize(cam, (28, 28))
        pred_cls = int(cls.numpy())

        axes[0, j].imshow(img.reshape(28, 28), cmap="gray")
        axes[0, j].set_title(f"輸入(正解 {want})", fontsize=9)
        axes[0, j].axis("off")
        axes[1, j].imshow(img.reshape(28, 28), cmap="gray")
        axes[1, j].imshow(cam, cmap="jet", alpha=0.5)
        axes[1, j].set_title(f"Grad-CAM(預測 {pred_cls})", fontsize=9)
        axes[1, j].axis("off")
    fig.suptitle("Grad-CAM:CNN 判斷每個數字時,注意力落在哪裡(紅=關鍵)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{out}/gradcam.png", dpi=150)
    print(f"已輸出:{out}/gradcam.png")


if __name__ == "__main__":
    main()
