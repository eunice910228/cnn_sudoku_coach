# -*- coding: utf-8 -*-
"""
photo_to_puzzle.py — 照片 → puzzle.txt(辨識這一步,單獨跑)
─────────────────────────────────────────────────────────────
為什麼分開:辨識需要 TensorFlow(建議 Python 3.12);而玩的 App(play_gui.py)
零依賴、任何 Python 都能跑。先用這支把照片變成 puzzle.txt,再到 App 用
「載入盤面檔…」開來玩——辨識在哪台跑都行,玩的人不必裝 TF。

會印出辨識結果,並標出『信心最低的可疑格』(白盒精神:先告訴你哪裡可能讀錯)。

執行:python app/photo_to_puzzle.py 照片.jpg [輸出.txt]
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from vision.recognize import (recognize_photo, grid_to_text,   # noqa: E402
                              suspect_cells)


def main():
    if len(sys.argv) < 2:
        sys.exit("用法:python app/photo_to_puzzle.py 照片.jpg [輸出.txt]")
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "puzzle.txt")

    try:
        grid, conf = recognize_photo(img_path)
    except Exception as e:
        sys.exit(f"辨識失敗:{e}")

    text = grid_to_text(grid)
    print("【辨識結果】")
    print(text)

    # 可疑格(信心低)— 提醒人工對照照片
    suspects = suspect_cells(grid, conf)
    avg = sum(conf[r][c] for r in range(9) for c in range(9)) / 81
    print(f"\n平均信心:{avg:.1%}")
    if suspects:
        print(f"⚠ 信心偏低、建議對照照片確認的格子({len(suspects)} 個):")
        for r, c in sorted(suspects, key=lambda rc: conf[rc[0]][rc[1]]):
            print(f"    R{r+1}C{c+1}: 讀成 {grid[r][c]}(信心 {conf[r][c]:.0%})")
    else:
        print("所有格子信心都高。")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 由 photo_to_puzzle.py 從照片辨識產生;請對照照片核對可疑格再用\n")
        f.write(text + "\n")
    print(f"\n已寫出:{out_path}")
    print("→ 到 App 按「載入盤面檔…」選這個檔,或先用編輯器把可疑格改對。")


if __name__ == "__main__":
    main()
