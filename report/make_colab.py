# -*- coding: utf-8 -*-
"""
make_colab.py — 產生可在 Google Colab 跑的 notebook(SudokuCoach_Colab.ipynb)
Colab 內建 TensorFlow,直接繞過本機安裝問題。涵蓋:訓練 → 實驗出圖 → 生成 PPTX。
(tkinter 桌面 App 在 Colab 無法跑,留在本機 3.14。)
執行:python report/make_colab.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    # nbformat 慣例:每行結尾保留 \n(最後一行不用)
    out = []
    for i, ln in enumerate(lines):
        out.append(ln + ("\n" if i < len(lines) - 1 else ""))
    return out


CELLS = [
    md("# 數獨教練 SudokuCoach — Colab 版",
       "",
       "在 Colab 跑 **訓練 → 實驗出圖 → 生成 PPTX**(Colab 內建 TensorFlow,免裝)。",
       "",
       "> ⚠️ 桌面 App(`app/play_gui.py`,tkinter)需要視窗,**Colab 不能跑**——那個留在本機(Python 3.14、零依賴)。",
       "> 建議把執行階段設成 GPU(編輯 → 筆記本設定 → GPU)會快一點,但 CPU 也能跑。"),

    md("## 步驟 1|環境設定(中文字型 + python-pptx)",
       "Colab 沒有微軟正黑,圖上的中文會變豆腐 → 裝 Noto CJK;PPTX 套件也一起裝。"),
    code("# 中文字型(避免圖變豆腐)+ python-pptx;TensorFlow 已內建",
         "!apt-get -qq install -y fonts-noto-cjk > /dev/null",
         "!pip -q install python-pptx",
         "!rm -rf ~/.cache/matplotlib   # 清字型快取,讓新字型被認出",
         "import tensorflow as tf",
         "print('環境就緒。TensorFlow', tf.__version__)"),

    md("## 步驟 2|從 GitHub 取得專案",
       "把 `REPO` 換成你的 **公開** repo 網址,執行就會 clone 下來(已經有了會改成 `git pull` 更新),再自動切進專案根。",
       "",
       "> 還沒推上 GitHub?在『有最新版的那台電腦』跑一次:`gh auth login` →",
       "> `gh repo create sudoku_coach --public --source=. --remote=origin --push`。public 才能讓 Colab 免認證 clone。",
       "> (不想用 GitHub 也行:zip 上傳或掛 Google Drive,見最後備註。)"),
    code("REPO = 'https://github.com/你的帳號/sudoku_coach.git'   # ← 改成你的",
         "import os",
         "name = REPO.rsplit('/', 1)[-1]",
         "name = name[:-4] if name.endswith('.git') else name",
         "path = '/content/' + name",
         "if os.path.isdir(path):",
         "    !cd \"$path\" && git pull          # 已有 → 更新到最新",
         "else:",
         "    !git clone \"$REPO\"               # 第一次 → 下載",
         "os.chdir(path)",
         "print('✅ 專案根:', os.getcwd())",
         "!ls"),

    md("## 步驟 3|訓練模型",
       "訓練幾個候選、挑對真實照片(`01.jpg`)最穩的存成 `vision/digit_cnn.h5`",
       "(含修掉空格幻覺數字的『邊框殘留增強』)。"),
    code("!python vision/restore_best.py"),

    md("## 步驟 4|跑實驗、產生所有圖",
       "域偏移 / 穩健性 / 消融 / Grad-CAM / t-SNE — 全部輸出到 `experiments/figures/`。"),
    code("!python experiments/eval_real.py",
         "!python experiments/robustness.py",
         "!python experiments/ablation_aug.py    # 會訓練兩個模型對照,稍久",
         "!python experiments/visualize_cnn.py",
         "!python experiments/extras.py"),

    md("## 步驟 5|做報告圖 + 生成 PPTX",
       "訓練曲線 / 混淆矩陣 / demo 示意,再把 15 頁簡報組出來。"),
    code("!python report/make_report_figs.py",
         "!python report/build_pptx.py"),

    md("## 步驟 6|看圖 + 下載 PPTX"),
    code("from IPython.display import Image, display",
         "import glob",
         "for f in sorted(glob.glob('experiments/figures/*.png')):",
         "    print(f); display(Image(f, width=640))"),
    code("from google.colab import files",
         "files.download('report/SudokuCoach.pptx')"),

    md("## 備註",
       "- **桌面 App**(`play_gui.py`)、`solve_grid.py`、`learn/lab.py` 屬零依賴,在**本機**跑就好(含 Python 3.14)。",
       "- 每次重開 Colab session 要重跑 **步驟 1~2**(環境不會留;但步驟 2 改 GitHub 後,clone/pull 一行就回來了)。",
       "- 改了程式想在 Colab 更新:在那台 push(`git add -A && git commit -m ... && git push`),Colab 重跑步驟 2 會自動 `git pull`。",
       "- **不用 GitHub 的替代**:",
       "  - 上傳 zip:`from google.colab import files; up=files.upload()` 後解壓再 `os.chdir`。",
       "  - 掛 Google Drive:`from google.colab import drive; drive.mount('/content/drive')` 後 `os.chdir('/content/drive/MyDrive/sudoku_coach')`。",
       "- 想自己學 CNN:`learn/lab.py` 也能在 Colab 用 `!python learn/lab.py --no-pause` 跑過一輪。"),
]

NB = {
    "cells": CELLS,
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(ROOT, "report", "SudokuCoach_Colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(NB, f, ensure_ascii=False, indent=1)
print("已產出 Colab 筆記本:", out)
print(f"共 {len(CELLS)} 個 cell")
