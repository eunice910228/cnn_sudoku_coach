# experiments/ — 深化 CNN 報告的實驗

把「我訓了一個 CNN」升級成「我理解這個 CNN 的能力與極限」。
每支腳本都會在 `experiments/figures/` 產出可直接貼進投影片的 PNG。

## 前置
```bash
pip install -r requirements.txt          # 需 matplotlib、scikit-learn
cd vision && python train.py             # 先有 vision/digit_cnn.h5
```
所有腳本從專案根目錄或 experiments/ 內執行都可以。

---

## 四個方向 → 跑法 → 對應投影片

### 方向一|Sim-to-Real 域偏移(報告的學術核心)
```bash
python experiments/eval_real.py      # 合成 vs 真實照片準確率
python experiments/robustness.py     # 拍照劣化壓力測試(落差真正現形處)
```
**自己跑出來的關鍵發現：**
- `01.jpg` 是乾淨數位截圖 → 模型 **100% 全對**,合成測試 99.75%,幾乎無落差。
- 壓力測試才看到極限:對焦模糊、感光雜訊、JPEG 壓縮都**很穩**;
  但**光線不均**讓準確率 100% → 76% → 69% 崩壞。
- 這正好解釋了拍照提示為何寫「**光線均勻**」,也指向未來改進方向
  (前處理做光照校正、或在資料增強裡加入光照梯度)。
- 圖:`real_vs_synth.png`、`grid_compare.png`、`robustness_curves.png`、`degradation_demo.png`
- → **Slide 6/9 之間插入「域偏移與穩健性」一頁**

### 方向二|資料增強消融(CNN 報告硬通貨)
```bash
python experiments/ablation_aug.py   # 訓練「有/無增強」兩個模型對照(約 3~5 分)
```
**自己跑出來的關鍵發現：**
- 有增強 vs 無增強,真實照片準確率 **100% vs 87.65%**(+12 個百分點);
  帶擾動的合成測試集 **98.65% vs 67.20%**(+31 個百分點)。
- 用數字證明「資料增強」這個設計決策不是裝飾,是穩健性的來源。
- 圖:`ablation_bar.png`、`ablation_robustness.png`
- → **Slide 9 訓練與評估,或單獨一頁「消融實驗」**

### 方向三|CNN 內部視覺化(吸睛王)
```bash
python experiments/visualize_cnn.py  # 濾波器 + 特徵圖 + Grad-CAM
```
- `filters.png`:第一層 32 個濾波器(模型自學的邊緣/筆畫偵測器)→ **Slide 8**
- `feature_maps.png`:一個真實數字觸發了哪些通道
- `gradcam.png`:Grad-CAM 熱力圖,顯示 CNN 判斷時「看哪裡」→ **Slide 8 加碼**

### 其他加碼
```bash
python experiments/extras.py         # t-SNE + 信心分布 × 矛盾偵測
```
- `tsne.png`:倒數第二層 128 維特徵壓到 2D,1~9 與空格自動分成 10 群
  → 證明學到的是可分離的語意表徵 → **收尾頁**
- `confidence.png`:81 格 softmax 信心分布 + 「白盒替黑盒把關」示意 → **Slide 12**

---

## 一句話總結這批實驗(可放結論頁)
> 模型在乾淨輸入上 100% 正確;壓力測試找出它唯一的弱點是光線不均;
> 消融實驗證明資料增強帶來 +12pp 的真實穩健性;
> 視覺化打開黑盒,看見它自學的濾波器與注意力。
> ——不只「會用 CNN」,而是「量得出它的能力邊界」。

## 注意
- 圖表上的中文需要系統有中文字型(Windows 內建微軟正黑體即可)。
- `labels/01.txt` 是 `01.jpg` 的人工正解(ground truth);換照片要另外建對應標籤檔。
