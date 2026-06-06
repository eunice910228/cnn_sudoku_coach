# -*- coding: utf-8 -*-
"""
play_gui.py — 數獨教練 桌面應用程式(視窗版,零依賴)
═══════════════════════════════════════════════════════════════════
互動模型(就是你說的那個):
    玩家自己填一格(任何順序都行)→ 按「下一步提示」
    → 程式讀『目前盤面』→ 只給下一步的思路 + 第一次用到的招式心法
    → 玩家自己填 → 再要提示……

關鍵:提示是『從你現在的盤面』即時推算的下一步,
      所以你用什麼順序填都無所謂,永遠對得上——不存在「思路對不齊」的問題。

只用 Python 內建的 tkinter,不需要 TensorFlow / OpenCV,任何 Python(含 3.14)都能跑。
執行:python app/play_gui.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from reasoning.board import Board                       # noqa: E402
from reasoning.solver import peek_next                  # noqa: E402
from reasoning.coach import TECHNIQUE_LESSONS           # noqa: E402

# 內建範例題(givens),0 = 空格
SAMPLE = [
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


# ───────────────────── 純邏輯(可單獨測試,不碰 GUI) ─────────────────────

def find_conflicts(grid):
    """回傳所有『違反數獨規則(同列/行/宮重複)』的格子座標集合——抓玩家填錯。"""
    bad = set()

    def scan(cells):
        seen = {}
        for (r, c) in cells:
            v = grid[r][c]
            if v == 0:
                continue
            if v in seen:
                bad.add((r, c))
                bad.add(seen[v])
            else:
                seen[v] = (r, c)

    for i in range(9):
        scan([(i, j) for j in range(9)])              # 列
        scan([(j, i) for j in range(9)])              # 行
    for b in range(9):
        br, bc = 3 * (b // 3), 3 * (b % 3)
        scan([(br + i, bc + j) for i in range(3) for j in range(3)])  # 宮
    return bad


def compute_hint(grid, taught):
    """讀『目前盤面』,回傳下一步該怎麼走。taught: 已教過的技巧集合(會就地更新)。

    回傳 dict 之一:
      {"conflict": {(r,c),...}}   玩家填錯了(有重複),先改對
      {"done": True}              已完成
      {"contradiction": (r,c)}    某空格已無候選 → 盤面湊不出解(常是前面填錯)
      {"stuck": True}             基礎四招用盡(需要更進階技巧)
      {"step": Step, "lesson": str|None, "is_new": bool}   下一步 + (新招)心法
    """
    conflicts = find_conflicts(grid)
    if conflicts:
        return {"conflict": conflicts}

    board = Board(grid)
    if board.is_solved():
        return {"done": True}

    step = peek_next(board)
    if step is None:
        bad = board.find_contradiction()
        if bad:
            return {"contradiction": bad}
        return {"stuck": True}

    lesson = None
    is_new = False
    if step.technique not in taught:
        lesson = TECHNIQUE_LESSONS.get(step.technique)
        taught.add(step.technique)
        is_new = lesson is not None
    return {"step": step, "lesson": lesson, "is_new": is_new}


# ───────────────────────────── GUI ─────────────────────────────

def launch():
    import tkinter as tk
    from tkinter import font as tkfont

    GIVEN_FG = "#1a1a1a"      # 題目給的數字(黑)
    USER_FG = "#1565C0"       # 玩家填的數字(藍)
    HINT_BG = "#FFF1A8"       # 提示高亮(黃)
    BAD_BG = "#F8C6C6"        # 錯誤高亮(紅)
    SUSPECT_BG = "#FFD9A8"    # 照片辨識可疑格(橘)
    OK_BG = "#FFFFFF"

    root = tk.Tk()
    root.title("數獨教練 SudokuCoach")
    root.configure(bg="#ECEFF1")

    big = tkfont.Font(family="Consolas", size=20)
    grid_state = {"given": [[False] * 9 for _ in range(9)]}
    taught = set()

    board_frame = tk.Frame(root, bg="#37474F", bd=2)
    board_frame.grid(row=0, column=0, padx=12, pady=12)

    cells = [[None] * 9 for _ in range(9)]
    vars_ = [[tk.StringVar() for _ in range(9)] for _ in range(9)]

    def make_validator():
        def vc(p):
            return p == "" or (len(p) == 1 and p in "123456789")
        return root.register(vc), "%P"

    vcmd = make_validator()

    for r in range(9):
        for c in range(9):
            e = tk.Entry(board_frame, width=2, font=big, justify="center",
                         textvariable=vars_[r][c], validate="key",
                         validatecommand=vcmd, relief="flat", bg=OK_BG,
                         disabledbackground=OK_BG, disabledforeground=GIVEN_FG)
            padx = (3 if c % 3 == 0 and c else 1, 1)
            pady = (3 if r % 3 == 0 and r else 1, 1)
            e.grid(row=r, column=c, padx=padx, pady=pady, ipady=4)
            cells[r][c] = e

    def read_grid():
        g = []
        for r in range(9):
            row = []
            for c in range(9):
                s = vars_[r][c].get()
                row.append(int(s) if s.isdigit() else 0)
            g.append(row)
        return g

    def clear_highlights():
        for r in range(9):
            for c in range(9):
                if not grid_state["given"][r][c]:
                    cells[r][c].configure(bg=OK_BG)

    def load(grid, lock_givens=True, msg_text=None):
        """lock_givens=True:非空格當成『題目』鎖定(黑色不可改)。
        lock_givens=False:全部可編輯——給照片辨識結果用,方便改掉讀錯的格子。"""
        taught.clear()
        for r in range(9):
            for c in range(9):
                v = grid[r][c]
                lock = lock_givens and v != 0
                grid_state["given"][r][c] = lock
                cells[r][c].configure(state="normal")
                vars_[r][c].set(str(v) if v != 0 else "")
                if lock:
                    cells[r][c].configure(fg=GIVEN_FG, state="disabled")
                else:
                    cells[r][c].configure(fg=USER_FG, bg=OK_BG)
        say(msg_text or "已載入題目。自己填一格,卡住就按「下一步提示」——"
            "提示永遠是『從你現在的盤面』算出來的下一步。")

    # 右側:提示 / 思路 文字區
    side = tk.Frame(root, bg="#ECEFF1")
    side.grid(row=0, column=1, sticky="n", padx=(0, 12), pady=12)

    msg = tk.Text(side, width=34, height=12, wrap="word", font=("Microsoft JhengHei", 10),
                  bg="#FFFFFF", relief="flat", padx=8, pady=8)
    msg.grid(row=0, column=0, columnspan=2, pady=(0, 8))
    msg.configure(state="disabled")

    def say(text, tag=None):
        msg.configure(state="normal")
        msg.delete("1.0", "end")
        msg.insert("1.0", text)
        msg.configure(state="disabled")

    def cell_name(r, c):
        return f"R{r + 1}C{c + 1}"

    def on_hint():
        clear_highlights()
        grid = read_grid()
        res = compute_hint(grid, taught)

        if "conflict" in res:
            for (r, c) in res["conflict"]:
                cells[r][c].configure(bg=BAD_BG)
            say("⚠ 這裡有重複的數字(紅色格子),違反數獨規則。\n"
                "先把填錯的改掉,我才能給你正確的下一步。")
            return
        if res.get("done"):
            say("🎉 全部完成,而且完全正確!你靠自己解開了。")
            return
        if res.get("contradiction"):
            r, c = res["contradiction"]
            cells[r][c].configure(bg=BAD_BG)
            say(f"⚠ {cell_name(r, c)} 已經沒有任何數字能填了——"
                "代表前面某一格填錯,導致湊不出解。回頭檢查一下。")
            return
        if res.get("stuck"):
            say("這題用基礎四招(唯一候選數 / 隱性唯一數 / 顯性數對 / 區塊摒除)"
                "已經推不下去了,需要更進階的技巧。先放著,或挑戰看看。")
            return

        step = res["step"]
        lines = []
        if step.kind == "place":
            r, c = step.cells[0]
            cells[r][c].configure(bg=HINT_BG)
            lines.append(f"【下一步】看 {cell_name(r, c)}(黃色格)")
        else:
            for (r, c) in step.cells:
                cells[r][c].configure(bg=HINT_BG)
            lines.append("【下一步】可以刪候選(黃色格)")
        lines.append(f"招式:{step.technique}")
        lines.append(f"思路:{step.reason}")
        if res["is_new"] and res["lesson"]:
            lines.append(f"\n🆕 第一次用到這招——心法:\n{res['lesson']}")
        lines.append("\n(自己把它填上去,再按一次要下一步)")
        say("\n".join(lines))

    def on_check():
        clear_highlights()
        grid = read_grid()
        conflicts = find_conflicts(grid)
        if conflicts:
            for (r, c) in conflicts:
                cells[r][c].configure(bg=BAD_BG)
            say(f"發現 {len(conflicts)} 個衝突格(紅色):同列/行/宮有重複數字。")
        else:
            board = Board(grid)
            bad = board.find_contradiction()
            if bad:
                cells[bad[0]][bad[1]].configure(bg=BAD_BG)
                say(f"目前沒有明顯重複,但 {cell_name(*bad)} 已無可填數字,"
                    "代表前面有一步填錯了。")
            elif board.is_solved():
                say("🎉 完成且正確!")
            else:
                say("目前沒有錯誤,繼續加油!")

    def on_clear_user():
        clear_highlights()
        for r in range(9):
            for c in range(9):
                if not grid_state["given"][r][c]:
                    vars_[r][c].set("")
        taught.clear()
        say("已清掉你填的部分,題目保留。")

    def on_blank():
        load([[0] * 9 for _ in range(9)])
        say("空白盤面。可以照著照片把題目打進去(自己當輸入),再開始解。")

    def on_photo():
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            title="選一張數獨照片",
            filetypes=[("圖片", "*.jpg *.jpeg *.png *.bmp"), ("所有檔案", "*.*")])
        if not path:
            return
        say("辨識中…(第一次載入模型會慢幾秒)")
        root.update()
        try:
            from vision.recognize import recognize_photo, suspect_cells
            grid, conf = recognize_photo(path)
        except ImportError:
            say("這個 Python 沒有裝 TensorFlow,無法在 App 內直接辨識。\n\n"
                "改用辨識專用環境(Python 3.12)跑:\n"
                "  python app/photo_to_puzzle.py 你的照片.jpg\n"
                "會產生 puzzle.txt,再用「載入盤面檔…」讀進來即可。")
            return
        except Exception as e:
            messagebox.showerror("辨識失敗", str(e))
            say("辨識失敗。可改用「載入盤面檔…」,或自己照著照片填。")
            return

        # 辨識結果『全部可編輯』(方便改掉讀錯的格子),可疑格標橘色
        load(grid, lock_givens=False,
             msg_text="已從照片辨識並填入(全部可改)。")
        suspects = suspect_cells(grid, conf)
        for (r, c) in suspects:
            cells[r][c].configure(bg=SUSPECT_BG)
        tip = ("📷 辨識完成。橘色格是『信心較低、建議對照照片確認』的格子"
               if suspects else "📷 辨識完成,信心都很高。")
        tip += ("\n\n感知(CNN)可能讀錯——這裡全部可編輯,對照照片改好,"
                "再按「下一步提示」開始解。\n"
                "(填錯造成矛盾時,「檢查對錯」會幫你抓出來。)")
        say(tip)

    def on_load_file():
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(
            title="選擇盤面檔(每列 9 個數字,空格用 0 或 .)",
            filetypes=[("文字盤面", "*.txt"), ("所有檔案", "*.*")])
        if not path:
            return
        digits = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                for ch in line.split("#", 1)[0]:        # 忽略註解
                    if ch.isdigit():
                        digits.append(int(ch))
                    elif ch == ".":
                        digits.append(0)                # 其餘(| + - 空白)跳過
        if len(digits) != 81:
            messagebox.showerror("格式不對",
                                 f"需要剛好 81 個格子,讀到 {len(digits)} 個。\n"
                                 "每列 9 個數字、共 9 列,空格用 0 或 .")
            return
        load([digits[r * 9:(r + 1) * 9] for r in range(9)])
        say(f"已從檔案載入盤面:{os.path.basename(path)}\n"
            "(照片辨識出來的 puzzle.txt 也能這樣讀進來。)")

    # 按鈕列
    btns = [
        ("下一步提示", on_hint, "#1565C0", "white"),
        ("檢查對錯", on_check, "#00897B", "white"),
        ("清空我填的", on_clear_user, "#90A4AE", "white"),
        ("從照片載入…", on_photo, "#6A1B9A", "white"),
        ("載入範例", lambda: load(SAMPLE), "#546E7A", "white"),
        ("載入盤面檔…", on_load_file, "#546E7A", "white"),
        ("空白盤面", on_blank, "#546E7A", "white"),
    ]
    for i, (text, cmd, bg, fg) in enumerate(btns):
        tk.Button(side, text=text, font=("Microsoft JhengHei", 10, "bold"),
                  bg=bg, fg=fg, relief="flat", width=14, command=cmd).grid(
            row=1 + i, column=0, columnspan=2, sticky="we", pady=2)

    def live_check(*_):
        """每打一個數字就即時檢查:造成重複的格子立刻變紅(警告,但不擋你填)。
        只管理紅色,不會蓋掉提示(黃)或可疑(橘)高亮。"""
        grid = read_grid()
        conflicts = find_conflicts(grid)
        for r in range(9):
            for c in range(9):
                if grid_state["given"][r][c]:
                    continue
                if (r, c) in conflicts:
                    cells[r][c].configure(bg=BAD_BG)
                elif str(cells[r][c].cget("bg")) == BAD_BG:
                    cells[r][c].configure(bg=OK_BG)     # 衝突解除 → 還原

    for r in range(9):
        for c in range(9):
            vars_[r][c].trace_add("write", live_check)

    load(SAMPLE)
    root.mainloop()


if __name__ == "__main__":
    launch()
