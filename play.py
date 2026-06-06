# -*- coding: utf-8 -*-
"""
play.py — 數獨教練 MVP(零依賴:python3 play.py)

架構(第一層設計):
  感知(CNN)只在「開局」出場一次,把題目搬進數位世界;
  之後盤面狀態存在程式裡,玩家在數位端遊玩,提示即時運算——
  教練不需要眼睛,因為桌子是它的。

用法:
  python3 play.py            # 選內建題目(簡單/中等/困難)
  python3 play.py photo.jpg  # 從照片開局(需安裝 vision 相關套件)

遊戲指令:
  3 5 7   → 在第3列第5行填入7
  h       → 跟教練要「下一步提示」(第一次用到的技巧會附教學)
  b       → 重看目前盤面
  q       → 投降,顯示完整解答
"""
import sys

from reasoning.board import Board
from reasoning.coach import CoachSession
from reasoning.solver import full_solution, solve_with_explanations, rate_difficulty


PUZZLES = {
    "1": ("簡單", "530070000600195000098000060800060003400803001"
                 "700020006060000280000419005000080079"),
    "2": ("中等", "300000000970010000600583000200000900500621003"
                 "008000005000435002000090056000000001"),
    "3": ("困難", "043080250600000000000001094900004070000608000"
                 "010200003820500000000000005034090710"),
}


def parse(s):
    return [[int(s[r * 9 + c]) for c in range(9)] for r in range(9)]


def load_from_photo(path):
    """感知模組的唯一出場時刻:把紙上的題目搬進數位世界(一次)。"""
    try:
        import numpy as np
        import cv2
        from tensorflow import keras
        from vision.grid_extractor import find_board, slice_cells
    except ImportError:
        sys.exit("照片開局需要 opencv / tensorflow,請先 pip install -r requirements.txt")

    image = cv2.imread(path)
    if image is None:
        sys.exit(f"讀不到圖片:{path}")
    board_img = find_board(image)
    if board_img is None:
        sys.exit("找不到數獨盤面(請拍正一點、光線均勻)")
    cells = slice_cells(board_img)
    model = keras.models.load_model("vision/digit_cnn.h5")
    digits = np.argmax(model.predict(cells, verbose=0), axis=1)
    return [[int(digits[r * 9 + c]) for c in range(9)] for r in range(9)]


def choose_puzzle():
    print("選擇題目難度:")
    for k, (name, _) in PUZZLES.items():
        print(f"  {k}. {name}")
    while True:
        c = input("> ").strip()
        if c in PUZZLES:
            return parse(PUZZLES[c][1])
        print("請輸入 1 / 2 / 3")


def main():
    # ---------- 開局:取得題目(照片或內建) ----------
    if len(sys.argv) == 2:
        grid = load_from_photo(sys.argv[1])
        print("【CNN 辨識結果】請確認是否正確:")
        print(Board(grid).pretty())
    else:
        grid = choose_puzzle()

    solution = full_solution(grid)
    if solution is None:
        sys.exit("⚠️ 此盤面無解——若來自照片,可能是辨識錯誤,請重拍。")

    # 開局時先評本題難度(用人類技巧走一遍)
    _, ref_steps = solve_with_explanations(grid)
    label, _ = rate_difficulty(ref_steps)

    coach = CoachSession(grid)
    board = coach.board
    givens = {(r, c) for r in range(9) for c in range(9) if grid[r][c] != 0}
    mistakes = hints_used = 0

    print(f"\n本題難度:{label}")
    print("我是你的數獨教練 🐱 我不會劇透,卡住再叫我(h)。\n")
    print(board.pretty())

    # ---------- 遊戲主迴圈(即時:狀態在記憶體,提示微秒級) ----------
    while not board.is_solved():
        try:
            cmd = input("\n[列 行 數字] / h 提示 / b 看盤 / q 投降 > ").strip().lower()
        except EOFError:
            print("\n(輸入結束)")
            return

        if cmd == "q":
            print("\n(完整解答)")
            print(Board(solution).pretty())
            print("\n下次再戰!卡住可以先用 h,教練只給一步,不會劇透整盤。")
            return

        if cmd == "b":
            print(board.pretty())
            continue

        if cmd == "h":
            hint = coach.next_hint()
            if hint is None:
                break
            if hint.get("stuck"):
                print("🤔 基礎技巧用盡了——這題進入進階區,要不要自己試試?(q 可看解答)")
                continue
            if hint.get("contradiction"):
                print("⚠️ 盤面有矛盾(若由照片而來,可能是辨識錯誤)。")
                return
            step = hint["step"]
            hints_used += 1
            if hint["is_new_technique"]:
                print(f"🆕 新招解鎖【{step.technique}】")
                print(f"   心法:{hint['lesson']}")
            print(f"💡 提示 ▶ {step.reason}")
            print("   (照著填填看——要自己動手,才算你的。)")
            continue

        # ---------- 玩家填數 ----------
        parts = cmd.split()
        if len(parts) != 3 or not all(p.isdigit() for p in parts):
            print("指令格式:列 行 數字(例如:3 5 7)")
            continue
        r, c, v = (int(p) for p in parts)
        if not (1 <= r <= 9 and 1 <= c <= 9 and 1 <= v <= 9):
            print("列/行/數字都要在 1~9 之間")
            continue
        r, c = r - 1, c - 1
        if (r, c) in givens or board.grid[r][c] != 0:
            print("這格已經有數字了")
            continue

        if solution[r][c] == v:
            board.place(r, c, v)
            print(f"✅ R{r + 1}C{c + 1} = {v}")
            print(board.pretty())
        else:
            mistakes += 1
            print(f"❌ R{r + 1}C{c + 1} 不是 {v} 喔,再想想(或按 h 要提示)")

    # ---------- 結算 ----------
    print("\n🎉 解開了!而且每一格都是你自己填的。")
    print(f"   用了 {hints_used} 次提示、失誤 {mistakes} 次")
    if coach.taught:
        print("   這一局你學會的招式:")
        for t in sorted(coach.taught):
            print(f"     ✓ {t}")


if __name__ == "__main__":
    main()
