# 數獨教練 SudokuCoach
## 從感知到推理:以數獨為例的神經符號系統實作
*A Neuro-Symbolic Approach: from Perception to Reasoning*

---

### 一句話定位
> **這不是「會解數獨的 AI」,是「陪你玩數獨的教練」。**
> 卡住時只提示下一步、說明理由,第一次用到某招還會教你怎麼用——
> 解題的樂趣,始終留給玩家。完整答案只在主動投降時才出現。

### 報告破題建議
> 「有人會問:這題丟給 ChatGPT 不就好了?——這份報告,就是這個問題的答案。」
>
> LLM 是機率模型,能生出**聽起來**有道理的解釋,但那是事後編的,可能跟真正的計算無關、甚至自信地出錯。
> 本系統的推理是**白盒**:理由不是「講出來的」,是「算出來的」——每一步都可驗證、保證正確。
> 數獨只是載具,真正示範的原則是:**有精確解的問題,別用機率的鎚子敲。**

---

## 系統架構(兩個半腦)

```
照片 ─► [感知模組 vision/]  ─► 81 個數字 ─► [推理模組 reasoning/] ─► 教練輸出
         CNN(神經,黑盒)                     技巧引擎(符號,白盒)     提示/教學/難度
         會「看」,說不出理由                  會「想」,每步有理由
```
- 神經網路會看不會講理;符號邏輯會講理不會看。本系統讓兩者各司其職。
- **加碼整合點**:推理模組偵測到盤面矛盾時,反過來替 CNN 的辨識結果把關(白盒驗證黑盒)。

## 模組對照（↔ 報告章節）

| 路徑 | 角色 | 報告章節 |
|---|---|---|
| `reasoning/board.py` | 盤面與候選數維護 | 推理引擎 |
| `reasoning/techniques.py` | 四種人類技巧（每步附中文理由） | 推理引擎 |
| `reasoning/solver.py` | 解題迴圈、難度評級、回溯備援 | 推理引擎 |
| `reasoning/coach.py` | 教練層:逐步提示 + 技巧教學（核心） | 核心理念 |
| `vision/grid_extractor.py` | 影像前處理:找盤面、透視校正、切 81 格 | 系統架構 |
| `vision/digit_dataset.py` | 合成印刷體數字資料集（字型渲染+擾動） | 資料與訓練 |
| `vision/digit_cnn.py` | 小型 CNN（LeNet 等級）模型定義 | CNN 原理 |
| `vision/train.py` | 訓練 + 準確率 + 混淆矩陣 | 實驗結果 |
| `app/main.py` | 完整管線:照片→辨識→推理→解說 | Live Demo |
| `app/play_gui.py` | **桌面 App:視窗版互動教練(tkinter,零依賴)** | Live Demo |
| `app/photo_to_puzzle.py` | 照片→puzzle.txt(辨識這步,標可疑格;需 TF) | Live Demo |
| `vision/recognize.py` | 照片→盤面+信心(App 與命列共用的辨識函式) | 系統架構 |
| `vision/restore_best.py` | 訓練多個候選、挑對真實照片最穩的存檔 | 資料與訓練 |
| `play.py` | **MVP:互動遊戲 + 教練(零依賴)** | Live Demo |
| `solve_grid.py` | 手打/載入盤面 → 解題+思路(零依賴,救急用) | Live Demo |
| `coach_demo.py` | 教練模式自動走查（零依賴） | Live Demo |
| `demo.py` | 引擎全功能展示（零依賴） | Live Demo |
| `learn/` | **動手教材:走完一輪就懂 CNN(lab/課本/提問卡)** | CNN 原理 |
| `experiments/` | 深化實驗:域偏移/消融/Grad-CAM/t-SNE(出圖) | 實驗結果 |

## 快速開始
```bash
# 0) 桌面 App(視窗版,推薦先玩):自己填、要提示就給下一步思路
python app/play_gui.py           # tkinter 內建,任何 Python(含 3.14)、零依賴
#    互動模型:你填一格(任何順序)→ 按「下一步提示」→ 讀『目前盤面』給下一步
#    按鈕:下一步提示 / 檢查對錯 / 從照片載入… / 載入範例 / 載入盤面檔… / 空白盤面
#    「從照片載入…」:當機器有 TensorFlow 時,直接拍照→辨識→填進格子(可疑格標橘色)
#    沒裝 TF 也沒關係 → 見下面 photo_to_puzzle,再用「載入盤面檔…」

# 0b) 照片 → 盤面檔(辨識這一步,需 TensorFlow,建議 Python 3.12)
python app/photo_to_puzzle.py 照片.jpg   # 產出 puzzle.txt + 標出信心低的可疑格
#    辨識在哪台跑都行;玩的 App 不必裝 TF。辨識→檔案→App,各司其職。

# 1) 手打/載入盤面直接解(救急:CNN 讀錯導致無解時用)
python solve_grid.py             # 讀 puzzle.txt → 解題+思路重播+難度(零依賴)

# 2) 終端機 MVP:互動數獨 + 內建教練(零依賴)
python play.py

# 3) 一起學 CNN(走完一輪就懂架構與應用)
python learn/lab.py              # 8 關動手實驗室;搭配 learn/LEARN_CNN.md、learn/QUIZ.md

# 4) 引擎全功能 / 教練自動走查(零依賴)
python demo.py
python coach_demo.py

# 5) 訓練 CNN + 深化實驗(需 TensorFlow,建議 Python 3.12)
cd vision && python train.py     # 產出 digit_cnn.h5
python experiments/eval_real.py  # 域偏移、消融、Grad-CAM… 見 experiments/README.md

# 6) 完整管線(拍一張數獨 → 辨識 → 推理 → 解說)
pip install -r requirements.txt
python app/main.py photo.jpg
```

## 已實作的人類技巧（由簡到難）
1. 唯一候選數 Naked Single（★）
2. 隱性唯一數 Hidden Single（★★）
3. 顯性數對 Naked Pair（★★★）
4. 區塊摒除 Pointing（★★★☆）
5. 回溯法備援——超出技巧範圍時誠實標記,可作為「限制與展望」素材

## 報告
完整的逐頁藍圖(含 Q&A 預備與上台檢查清單)見 **REPORT.md**。

## 建議報告流程(CNN 深度版,~15 頁)
破題（「丟給 ChatGPT 不就好?」）→ 神經符號架構(背景)
→ **CNN 主菜**:前處理/為何不用 MNIST → CNN 架構 → 濾波器+特徵圖 → Grad-CAM
→ 混淆矩陣+t-SNE → **Sim-to-Real 穩健性(光線是唯一弱點)** → **資料增強消融(+12pp)**
→ **實戰除錯:白盒抓黑盒、把空格幻覺修在源頭**
→ 推理引擎(白盒) → Live Demo(桌面 App 為主、CLI 備援) → 限制與展望 → 結論
> 各頁的「⚠️自備」圖大多已備好,在 `experiments/figures/`;跑法見 `experiments/README.md`。
