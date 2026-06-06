# -*- coding: utf-8 -*-
"""
solve_grid.py — 手打盤面 → 解題 + 思路重播 + 難度(純 Python,零依賴)
═══════════════════════════════════════════════════════════════════
不需要 TensorFlow、不需要 OpenCV,任何 Python(含 3.14)都能跑。
當 CNN 把照片讀錯導致『無解』時,用這支:照著照片把盤面打對,直接解。

用法:
    python solve_grid.py              # 讀 puzzle.txt
    python solve_grid.py 我的題目.txt  # 讀指定檔案

盤面檔格式:9 列、每列 9 個數字,空格用 0 或 .;
            『#』開頭的行、還有 +-| 這類畫格線的符號都會被自動忽略。
"""
import os
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from reasoning.board import Board                                     # noqa: E402
from reasoning.solver import solve_with_explanations, rate_difficulty  # noqa: E402


def parse_grid(path):
    """把盤面檔解析成 9x9 的 list;空格=0。容錯:忽略註解與格線符號。"""
    digits = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0]          # 去掉註解
            for ch in line:
                if ch.isdigit():
                    digits.append(int(ch))
                elif ch == ".":
                    digits.append(0)
                # 其他字元(空白、| + - 等)一律跳過
    if len(digits) != 81:
        sys.exit(f"盤面需要剛好 81 個格子,但讀到 {len(digits)} 個。\n"
                 f"檢查 {path}:每列 9 個數字、共 9 列,空格用 0 或 .")
    return [digits[r * 9:(r + 1) * 9] for r in range(9)]


def basic_valid(grid):
    """先檢查輸入本身有沒有明顯矛盾(同列/行/宮重複),給出友善提示"""
    def dup(vals):
        v = [x for x in vals if x != 0]
        return len(v) != len(set(v))
    for i in range(9):
        if dup(grid[i]):
            return f"第 {i+1} 列有重複數字"
        if dup([grid[r][i] for r in range(9)]):
            return f"第 {i+1} 行有重複數字"
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):
            box = [grid[br+dr][bc+dc] for dr in range(3) for dc in range(3)]
            if dup(box):
                return f"第 {br//3*3+bc//3+1} 宮有重複數字"
    return None


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "puzzle.txt")
    if not os.path.exists(path):
        sys.exit(f"找不到盤面檔:{path}")

    grid = parse_grid(path)
    print(f"【讀入的盤面】來源:{os.path.basename(path)}")
    print(Board(grid).pretty())

    problem = basic_valid(grid)
    if problem:
        print(f"\n⚠ 輸入本身就有矛盾:{problem}")
        print("  → 這通常代表某格打錯了,對照照片改一下再跑。")
        return

    solved, steps = solve_with_explanations(grid)
    print("\n【思路重播】")
    for i, s in enumerate(steps, 1):
        print(f"  {i:>2}. [{s.technique}] {s.reason}")

    unsolved = any(0 in row for row in solved.grid) \
        if hasattr(solved, "grid") else None
    print("\n【解答】")
    print(solved.pretty())

    if any("無解" in s.technique for s in steps):
        print("\n⚠ 這盤『無解』——合法數獨一定有解,代表還有格子打錯了。")
        print("  先檢查:整列/整行全空的地方、還有信心低的格子。對照照片修正再跑一次。")
    else:
        label, _ = rate_difficulty(steps)
        print(f"\n【難度評級】{label}")


if __name__ == "__main__":
    main()
