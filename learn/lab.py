# -*- coding: utf-8 -*-
"""
lab.py — CNN 動手實驗室:走完一輪就懂整條流程
═══════════════════════════════════════════════════════════════════
一份引導式的動手課程。每一關先講白話、再讓你親眼看到結果。
順序就是 CNN 的完整生命週期:

  1) 資料長什麼樣        影像其實只是一堆數字
  2) 卷積(手算)        濾波器滑過影像 = 加權和,抓出特徵
  3) ReLU + 池化(手算)  留強反應、縮小、獲得平移容忍
  4) 堆成 CNN           把上面的積木疊起來,看每層形狀怎麼變
  5) 訓練              讓濾波器「自己長出來」,看 loss 下降
  6) 預測一格          softmax 機率分布,取最大就是答案
  7) 看它學到什麼       把第一層濾波器畫出來
  8) 接回完整應用       這顆 CNN 如何插進數獨教練管線

執行(會在每關之間停下來等你按 Enter,慢慢讀):
    python learn/lab.py
一口氣跑完不暫停:
    python learn/lab.py --no-pause
只跑某一關:
    python learn/lab.py 2          (只跑第 2 關「卷積」)
"""
import os
import sys

import numpy as np

# 讓中文不亂碼 + 找得到 vision 套件
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
FIG = os.path.join(ROOT, "learn", "figures")

NO_PAUSE = "--no-pause" in sys.argv


# ───────────────────────── 小工具 ─────────────────────────
def title(n, name):
    print("\n" + "═" * 64)
    print(f"  第 {n} 關|{name}")
    print("═" * 64)


def pause():
    """引導節奏:互動執行時等你按 Enter;被管線呼叫(非互動)時自動略過"""
    if NO_PAUSE or not sys.stdin.isatty():
        return
    try:
        input("\n  ── 按 Enter 繼續 ──")
    except EOFError:
        pass


def ascii_art(img, threshold_chars=" .:-=+*#%@"):
    """把 0~1 的小圖印成 ASCII,讓人『看見影像就是數字』"""
    img = np.asarray(img).reshape(img.shape[-2], img.shape[-1]) \
        if img.ndim > 2 else np.asarray(img)
    lo, hi = img.min(), img.max()
    norm = (img - lo) / (hi - lo + 1e-8)
    lines = []
    for row in norm:
        chars = [threshold_chars[min(len(threshold_chars) - 1,
                 int(v * (len(threshold_chars) - 1)))] for v in row]
        lines.append("".join(c * 2 for c in chars))   # *2 讓字寬接近正方
    return "\n".join(lines)


def maybe_show(fig, path):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(f"  🖼  已存圖:{path}")
    if sys.stdin.isatty() and not NO_PAUSE:
        try:
            import matplotlib.pyplot as plt
            plt.show()
        except Exception:
            pass


def get_one_digit(want=7):
    """從合成資料拿一張乾淨的數字小圖(28x28, 0~1)"""
    from vision.digit_dataset import make_dataset
    X, y = make_dataset(per_class=5, seed=0, augment=False)
    idx = np.where(y == want)[0]
    return X[idx[0]].reshape(28, 28)


# ───────────────────────── 第 1 關 ─────────────────────────
def stage1():
    title(1, "資料長什麼樣 — 影像其實只是一堆數字")
    print("""
  CNN 不會『看到』數字,它只看到一個矩陣。我們這份專案的每個格子是
  28×28 的灰階圖:每個像素是一個 0~1 的數(0=黑、1=白)。
  先把一個『7』印出來,左邊是亮度、右邊用符號讓你看出形狀:""")
    img = get_one_digit(7)
    print(f"\n  形狀 shape = {img.shape},數值範圍 {img.min():.2f} ~ {img.max():.2f}")
    print("\n  中間一列的原始數字(只取前 14 個):")
    mid = img[14]
    print("   ", " ".join(f"{v:.1f}" for v in mid[:14]))
    print("\n  整張圖的 ASCII 形狀:\n")
    print(ascii_art(img))
    print("""
  ☞ 重點:對電腦來說,『看圖』= 對這個數字矩陣做數學運算。
     接下來每一關,都是在這個矩陣上動手。""")
    pause()
    return img


# ───────────────────────── 第 2 關 ─────────────────────────
def conv2d_by_hand(img, kernel):
    """手刻卷積:濾波器在影像上每滑一格,做一次『對應相乘再加總』"""
    kh, kw = kernel.shape
    H, W = img.shape
    out = np.zeros((H - kh + 1, W - kw + 1))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = img[i:i + kh, j:j + kw]      # 蓋住的那一小塊
            out[i, j] = np.sum(patch * kernel)   # 加權和 ← 卷積的本質
    return out


def stage2(img=None):
    title(2, "卷積(手算)— 濾波器滑過影像,抓出特徵")
    if img is None:
        img = get_one_digit(7)
    print("""
  『濾波器(filter / kernel)』是一個小矩陣(這裡用 3×3)。它在大圖上
  一格一格滑動,每停一處就把蓋住的 3×3 區塊『對應相乘、全部加總』成
  一個數。掃完整張圖,就得到一張『特徵圖(feature map)』。

  不同的濾波器抓不同東西。我們手動設計兩個經典的:""")
    vert = np.array([[-1, 0, 1],
                     [-1, 0, 1],
                     [-1, 0, 1]], dtype="float32")   # 偵測『垂直邊』
    horiz = np.array([[-1, -1, -1],
                      [0,  0,  0],
                      [1,  1,  1]], dtype="float32")  # 偵測『水平邊』
    print("\n  垂直邊濾波器:\n", vert.astype(int))
    print("\n  水平邊濾波器:\n", horiz.astype(int))

    # 示範『一次滑動』的算術,讓人看見乘加
    patch = img[13:16, 13:16]
    print("\n  舉例:濾波器蓋在影像 (13,13) 這塊 3×3 上:")
    print("   影像區塊 =\n", np.round(patch, 2))
    print(f"   對應相乘再加總 = {np.sum(patch * vert):.2f}  ← 這就是特徵圖上的一個點")

    fv = conv2d_by_hand(img, vert)
    fh = conv2d_by_hand(img, horiz)
    print(f"\n  整張掃完:輸入 {img.shape} → 特徵圖 {fv.shape}(邊緣少 2 是正常的)")
    print("\n  『垂直邊』特徵圖(亮處 = 偵測到垂直筆畫):\n")
    print(ascii_art(np.abs(fv)))

    try:
        import matplotlib.pyplot as plt
        from _labutil import use_cjk
        use_cjk()
        fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
        ax[0].imshow(img, cmap="gray"); ax[0].set_title("輸入(7)")
        ax[1].imshow(np.abs(fv), cmap="magma"); ax[1].set_title("垂直邊濾波器")
        ax[2].imshow(np.abs(fh), cmap="magma"); ax[2].set_title("水平邊濾波器")
        for a in ax:
            a.axis("off")
        fig.suptitle("同一張圖、不同濾波器 → 抓出不同特徵", fontsize=13)
        fig.tight_layout()
        maybe_show(fig, os.path.join(FIG, "01_convolution.png"))
    except Exception as e:
        print(f"  (略過畫圖:{e})")
    print("""
  ☞ 重點:Conv2D 做的就是這件事——只是它一層有『32 個』濾波器,
     而且濾波器裡的數字不是我們設計的,是『訓練時自己學出來的』(第 5 關)。""")
    pause()
    return fv


# ───────────────────────── 第 3 關 ─────────────────────────
def relu_by_hand(x):
    return np.maximum(0, x)            # 負的全部變 0


def maxpool_by_hand(x, size=2):
    H, W = x.shape
    out = np.zeros((H // size, W // size))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            block = x[i*size:(i+1)*size, j*size:(j+1)*size]
            out[i, j] = block.max()    # 取區塊最大值
    return out


def stage3(fmap=None):
    title(3, "ReLU + 池化(手算)— 留強反應、縮小、容忍位移")
    if fmap is None:
        fmap = stage2_quiet()
    print("""
  卷積後通常接兩個動作:

  ① ReLU:把所有負數變成 0。直覺:『有偵測到特徵(正)就留著,
     沒偵測到(負)就當沒看到』,讓網路只保留有意義的強反應。

  ② 最大池化 MaxPooling:把圖切成 2×2 小塊,每塊只留『最大值』。
     效果:尺寸減半(運算變少)+ 數字稍微平移也不影響結果(平移容忍)。""")
    r = relu_by_hand(fmap)
    p = maxpool_by_hand(r, 2)
    print(f"\n  形狀變化:特徵圖 {fmap.shape} ──ReLU──> {r.shape} "
          f"──MaxPool2×2──> {p.shape}")
    print("\n  池化後的圖(變小了,但主要特徵還在):\n")
    print(ascii_art(p))
    print("""
  ☞ 重點:Conv → ReLU → Pool 是 CNN 最基本的『一個積木』。
     疊很多層,就能從『邊』組合出『部件』,再組合出『整個數字』。""")
    pause()


# ───────────────────────── 第 4 關 ─────────────────────────
def stage4():
    title(4, "堆成 CNN — 把積木疊起來,看每層形狀怎麼變")
    print("""
  現在看我們專案真正用的模型(vision/digit_cnn.py)。它就是把前三關的
  積木疊起來,最後接一個分類器。重點是『形狀(shape)怎麼一路變化』:""")
    from vision.digit_cnn import build_model
    model = build_model()
    model.summary(print_fn=lambda s: print("   " + s))
    print("""
  逐層白話解讀(形狀變化):
    輸入 28×28×1
    → Conv(32, 3×3) :用 32 個濾波器掃 → 26×26×32(32 張特徵圖)
    → MaxPool 2×2   :尺寸減半 → 13×13×32
    → Conv(64, 3×3) :更深、更多濾波器,組合出『部件』→ 11×11×64
    → MaxPool 2×2   :→ 5×5×64
    → Flatten       :把 5×5×64=1600 攤平成一條向量
    → Dense(128)    :全連接,綜合判斷
    → Dropout(0.3)  :訓練時隨機關掉 30% 神經元,防止死背(過擬合)
    → Dense(10)+softmax:輸出 10 個類別的機率(0=空格, 1~9)

  ☞ 重點:前半段(Conv/Pool)負責『抽特徵』,後半段(Dense)負責『做決定』。
     參數量大多在 Dense——這也是為什麼 Conv 很省、卻很會看。""")
    pause()
    return model


# ───────────────────────── 第 5 關 ─────────────────────────
def stage5():
    title(5, "訓練 — 讓濾波器『自己長出來』,看 loss 下降")
    print("""
  前面的垂直/水平邊濾波器是我們『手設計』的。真正的 CNN 不靠人設計——
  它一開始濾波器全是『隨機數』,然後反覆做這件事:

    看一批圖 → 猜答案 → 跟正解比,算出『錯多少』(loss)
    → 用梯度下降,把所有濾波器和權重往『錯更少』的方向微調一點點

  重複幾千次,濾波器就自己長成了邊緣/筆畫偵測器。我們現在用一小份
  資料、跑 3 輪(epoch),親眼看 loss 變小、準確率上升:""")
    from vision.digit_dataset import make_dataset
    from vision.digit_cnn import build_model
    print("\n  生成小型訓練資料中(每類 300 張)...")
    X, y = make_dataset(per_class=300, seed=1, augment=True)
    n = int(len(X) * 0.9)
    model = build_model()
    print("  開始訓練(看 accuracy 一輪一輪上升):\n")
    model.fit(X[:n], y[:n], epochs=3, batch_size=128,
              validation_split=0.1, verbose=2)
    acc = (np.argmax(model.predict(X[n:], verbose=0), axis=1) == y[n:]).mean()
    print(f"\n  測試準確率:{acc:.2%}")
    print("""
  ☞ 重點:訓練 = 自動調參數讓 loss 變小。沒有人告訴它『7 長怎樣』,
     它從上千個例子裡『歸納』出來。這就是『學習』。""")
    pause()
    return model


# ───────────────────────── 第 6 關 ─────────────────────────
def stage6(model):
    title(6, "預測一格 — softmax 機率分布,取最大就是答案")
    print("""
  訓練好後,餵一張圖進去,模型吐出 10 個數字——每個類別的『機率』
  (加起來=1,這叫 softmax)。最大的那個就是它的答案,該機率就是『信心』。""")
    img = get_one_digit(7)
    probs = model.predict(img.reshape(1, 28, 28, 1), verbose=0)[0]
    pred = int(np.argmax(probs))
    print(f"\n  餵進一張『7』,模型輸出的 10 類機率:\n")
    for k, p in enumerate(probs):
        bar = "█" * int(p * 40)
        label = "空格" if k == 0 else str(k)
        mark = "  ← 預測" if k == pred else ""
        print(f"    類別 {label:>2}: {p:6.2%} {bar}{mark}")
    print(f"\n  最終答案:{pred}(信心 {probs[pred]:.1%})")
    print("""
  ☞ 重點:輸出不是『一個答案』,而是『一份機率分布』。信心低的時候,
     就是模型在說『我不太確定』——這正是後面推理引擎要把關的地方。""")
    pause()


# ───────────────────────── 第 7 關 ─────────────────────────
def stage7(model):
    title(7, "看它學到什麼 — 把第一層濾波器畫出來")
    print("""
  我們在第 2 關『手設計』了垂直/水平邊濾波器。現在把第 5 關訓練出來的
  模型,第一層 32 個『自己學到』的濾波器畫出來——你會看到它長出了
  類似邊緣/筆畫的偵測器,沒有人教它。""")
    w = None
    for layer in model.layers:
        if "conv" in layer.name.lower():
            w = layer.get_weights()[0]       # (3,3,1,32)
            break
    try:
        import matplotlib.pyplot as plt
        from _labutil import use_cjk
        use_cjk()
        fig, axes = plt.subplots(4, 8, figsize=(9, 4.6))
        for i, ax in enumerate(axes.ravel()):
            if i < w.shape[-1]:
                f = w[:, :, 0, i]
                f = (f - f.min()) / (np.ptp(f) + 1e-8)
                ax.imshow(f, cmap="gray")
            ax.axis("off")
        fig.suptitle("第一層 32 個濾波器(模型訓練後自己長出來的)", fontsize=13)
        fig.tight_layout()
        maybe_show(fig, os.path.join(FIG, "02_learned_filters.png"))
    except Exception as e:
        print(f"  (略過畫圖:{e})")
    print("""
  ☞ 重點:這是 CNN 最迷人的地方——特徵不是工程師寫死的,是從資料
     『長出來』的。想看更深入的(特徵圖、Grad-CAM、t-SNE),
     去跑 experiments/ 裡的腳本。""")
    pause()


# ───────────────────────── 第 8 關 ─────────────────────────
def stage8():
    title(8, "接回完整應用 — 這顆 CNN 如何插進數獨教練")
    print("""
  恭喜,你已經走完一顆 CNN 的完整生命週期。最後看它在真實系統裡的位置:

    一張數獨照片
      │
      ├─【vision/grid_extractor.py】找盤面 → 透視校正 → 切成 81 個 28×28 小格
      │
      ├─【這顆 CNN】對每一格做第 6 關的預測 → 得到 81 個數字(0~9)
      │      (第 1~7 關學的,全部發生在這一步,重複 81 次)
      │
      └─【reasoning/】符號推理引擎邏輯解題、解釋每一步、評難度

  也就是說:
    • CNN 負責『看』(把紙上的數字變成電腦懂的數字)——會看,但說不出理由。
    • 推理引擎負責『想』(用數獨規則一步步推)——每步可解釋、保證正確。
    這個組合叫『神經符號(Neuro-Symbolic)』架構。

  想完整跑一次真實照片:
    python app/main.py 01.jpg
  想深入 CNN 的能力與極限(域偏移、消融、Grad-CAM):
    看 experiments/ 資料夾

  ☞ 你現在不只是『會用』,而是知道了:
     資料是什麼、卷積在算什麼、為什麼要池化、模型怎麼堆、
     訓練在調什麼、預測怎麼出來、它到底學到了什麼,以及這一切如何變成應用。""")
    pause()


# 給 stage3 在單獨執行時用的安靜版 stage2
def stage2_quiet():
    img = get_one_digit(7)
    vert = np.array([[-1, 0, 1]] * 3, dtype="float32")
    return conv2d_by_hand(img, vert)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args and args[0].isdigit():
        n = int(args[0])
        model = None
        if n == 1: stage1()
        elif n == 2: stage2()
        elif n == 3: stage3()
        elif n == 4: stage4()
        elif n in (5, 6, 7):
            model = stage5()
            if n >= 6: stage6(model)
            if n >= 7: stage7(model)
        elif n == 8: stage8()
        else: print(f"沒有第 {n} 關(共 1~8 關)")
        return

    print("""
  ┌──────────────────────────────────────────────────────────
  │  CNN 動手實驗室 — 走完一輪就懂整條流程
  │  讀說明 → 看結果 → 按 Enter,一關一關慢慢走
  └──────────────────────────────────────────────────────────""")
    img = stage1()
    fmap = stage2(img)
    stage3(fmap)
    stage4()
    model = stage5()
    stage6(model)
    stage7(model)
    stage8()
    print("\n  🎓 全部走完了。回頭重跑任何一關:python learn/lab.py <關號>\n")


if __name__ == "__main__":
    main()
