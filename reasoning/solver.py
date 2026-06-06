# -*- coding: utf-8 -*-
"""
solver.py — 推理引擎主迴圈
1. 依「由簡到難」嘗試人類技巧,每一步都留下中文理由(思路重播)。
2. 卡住時 fallback 到回溯法,並誠實標記「超出人類技巧範圍」。
3. 依使用到的最難技巧,給出整題難度評級。
4. hint(): 教練模式 — 只回傳「下一步」。
"""

from .board import Board
from .techniques import TECHNIQUES, Step


# ---------- 回溯法(備援,黑盒的暴力解) ----------

def _backtrack(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for v in range(1, 10):
                    if _valid(grid, r, c, v):
                        grid[r][c] = v
                        if _backtrack(grid):
                            return True
                        grid[r][c] = 0
                return False
    return True


def _valid(grid, r, c, v):
    if v in grid[r]:
        return False
    if any(grid[i][c] == v for i in range(9)):
        return False
    br, bc = 3 * (r // 3), 3 * (c // 3)
    return all(grid[i][j] != v for i in range(br, br + 3) for j in range(bc, bc + 3))


# ---------- 主流程 ----------

def _apply(board, step):
    if step.kind == "place":
        (r, c), v = step.cells[0], step.value
        board.place(r, c, v)
    else:
        for (r, c), vals in step.eliminations:
            board.candidates[r][c] -= vals


def solve_with_explanations(grid):
    """回傳 (board, steps):解完的盤面 + 完整思路鏈"""
    board = Board(grid)
    steps = []

    while not board.is_solved():
        # 安全閥:偵測矛盾(常見來源:CNN 認錯數字)
        bad = board.find_contradiction()
        if bad:
            r, c = bad
            steps.append(Step("矛盾偵測", "halt", [bad], None,
                              f"R{r + 1}C{c + 1} 已無任何候選數,盤面矛盾——"
                              f"推理模組判定輸入有誤(可能是影像辨識錯誤),中止解題", 0))
            return board, steps

        for tech in TECHNIQUES:
            step = tech(board)
            if step:
                break
        else:
            # 人類技巧全數用盡 → 回溯法補完
            snapshot = [row[:] for row in board.grid]
            if _backtrack(snapshot):
                steps.append(Step("回溯法 (Backtracking)", "place", [], None,
                                  "剩餘步驟超出目前實作的人類技巧範圍,"
                                  "改用回溯法(試誤+復原)補完剩餘格子", 5.0))
                board = Board(snapshot)
            else:
                steps.append(Step("無解", "halt", [], None, "此盤面無解", 0))
            return board, steps

        _apply(board, step)
        steps.append(step)

    return board, steps


def hint(grid):
    """教練模式:只給下一步(不破壞玩家的解題樂趣)"""
    board = Board(grid)
    if board.find_contradiction():
        return None
    for tech in TECHNIQUES:
        step = tech(board)
        if step:
            return step
    return None


# ---------- 難度評級 ----------

RATING = [
    (1.0, "★ 入門(只需唯一候選數)"),
    (2.0, "★★ 簡單(需要隱性唯一數)"),
    (3.0, "★★★ 中等(需要數對推理)"),
    (3.5, "★★★☆ 中等偏難(需要區塊摒除)"),
    (5.0, "★★★★★ 困難(超出基礎技巧,需試誤)"),
]


def rate_difficulty(steps):
    hardest = max((s.difficulty for s in steps), default=0)
    label = RATING[0][1]
    for d, text in RATING:
        if hardest >= d:
            label = text
    used = {}
    for s in steps:
        used[s.technique] = used.get(s.technique, 0) + 1
    return label, used


# ---------- 對外公開:套用單一步驟(供教練模式逐步推進) ----------

def apply_step(board, step):
    """把一個 Step 套用到 board 上(教練模式逐步使用)"""
    _apply(board, step)


def peek_next(board):
    """看 board 目前的下一步(不修改 board);回傳 Step 或 None"""
    if board.find_contradiction():
        return None
    for tech in TECHNIQUES:
        step = tech(board)
        if step:
            return step
    return None


def full_solution(grid):
    """回傳完整解(9x9 list);無解回傳 None。供遊戲端驗證玩家填數用。"""
    snapshot = [row[:] for row in grid]
    if _backtrack(snapshot):
        return snapshot
    return None
