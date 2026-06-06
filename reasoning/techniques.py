# -*- coding: utf-8 -*-
"""
techniques.py — 人類解題技巧模組(符號推理的核心)
每個技巧函式:吃進 Board,找到「一步」就回傳 Step(含中文理由);找不到回傳 None。
這就是『白盒』:每一步都有名字、有理由、可驗證。
"""

from dataclasses import dataclass, field


@dataclass
class Step:
    technique: str          # 技巧名稱
    kind: str               # "place"(填數) 或 "eliminate"(刪候選)
    cells: list             # 涉及的格子 [(r, c), ...]
    value: object           # 填入的數字,或被刪除的候選數元組
    reason: str             # 給人看的中文推理說明
    difficulty: float       # 技巧難度權重(用於整題難度評級)
    eliminations: list = field(default_factory=list)  # [((r,c), {被刪的候選}), ...]


def _cell(r, c):
    return f"R{r + 1}C{c + 1}"


# ---------------------------------------------------------------
# 技巧 1:唯一候選數 (Naked Single) — 難度 1.0
# 某格的候選數只剩一個 → 直接填。
# ---------------------------------------------------------------
def naked_single(board):
    for r in range(9):
        for c in range(9):
            if board.grid[r][c] == 0 and len(board.candidates[r][c]) == 1:
                v = next(iter(board.candidates[r][c]))
                reason = (f"{_cell(r, c)} 的候選數只剩 {v}"
                          f"(其餘 1~9 已出現在同列、同行或同宮)→ 填入 {v}")
                return Step("唯一候選數 (Naked Single)", "place",
                            [(r, c)], v, reason, 1.0)
    return None


# ---------------------------------------------------------------
# 技巧 2:隱性唯一數 (Hidden Single) — 難度 2.0
# 在某個列/行/宮裡,數字 v 只剩一個可放的位置 → 填。
# ---------------------------------------------------------------
def hidden_single(board):
    for name, cells in board.units():
        for v in range(1, 10):
            spots = [(r, c) for (r, c) in cells
                     if board.grid[r][c] == 0 and v in board.candidates[r][c]]
            if len(spots) == 1:
                r, c = spots[0]
                if len(board.candidates[r][c]) == 1:
                    continue  # 留給 Naked Single,理由更直觀
                reason = (f"在{name}中,數字 {v} 只剩 {_cell(r, c)} 一個可放位置"
                          f"(該數在其餘空格都已被同列/行/宮排除)→ 填入 {v}")
                return Step("隱性唯一數 (Hidden Single)", "place",
                            [(r, c)], v, reason, 2.0)
    return None


# ---------------------------------------------------------------
# 技巧 3:顯性數對 (Naked Pair) — 難度 3.0
# 同單位中兩格的候選數是相同的一對 {a, b} → 這對數必落在這兩格,
# 同單位其他格可刪除 a、b。
# ---------------------------------------------------------------
def naked_pair(board):
    for name, cells in board.units():
        seen = {}
        for (r, c) in cells:
            if board.grid[r][c] == 0 and len(board.candidates[r][c]) == 2:
                key = tuple(sorted(board.candidates[r][c]))
                seen.setdefault(key, []).append((r, c))
        for key, locs in seen.items():
            if len(locs) != 2:
                continue
            elims = []
            for (r, c) in cells:
                if (r, c) in locs or board.grid[r][c] != 0:
                    continue
                hit = board.candidates[r][c] & set(key)
                if hit:
                    elims.append(((r, c), hit))
            if elims:
                a, b = locs
                targets = "、".join(_cell(r, c) for (r, c), _ in elims)
                reason = (f"{name}中,{_cell(*a)} 與 {_cell(*b)} 的候選數同為 {set(key)}:"
                          f"這兩個數字必定分佔這兩格 → 從 {targets} 的候選中刪除 {set(key)}")
                return Step("顯性數對 (Naked Pair)", "eliminate",
                            [p for p, _ in elims], key, reason, 3.0, elims)
    return None


# ---------------------------------------------------------------
# 技巧 4:區塊摒除 (Pointing Pair / Box-Line Reduction) — 難度 3.5
# 某宮中,數字 v 的候選位置全部落在同一列(或同一行)
# → 該列(行)在這個宮以外的格子,都可刪除候選 v。
# ---------------------------------------------------------------
def pointing(board):
    for b in range(9):
        br, bc = 3 * (b // 3), 3 * (b % 3)
        box = [(br + i, bc + j) for i in range(3) for j in range(3)]
        for v in range(1, 10):
            spots = [(r, c) for (r, c) in box
                     if board.grid[r][c] == 0 and v in board.candidates[r][c]]
            if len(spots) < 2:
                continue
            rows = {r for r, _ in spots}
            cols = {c for _, c in spots}
            line = None
            if len(rows) == 1:
                rr = rows.pop()
                line = ("列", [(rr, c) for c in range(9) if (rr, c) not in box])
            elif len(cols) == 1:
                cc = cols.pop()
                line = ("行", [(r, cc) for r in range(9) if (r, cc) not in box])
            if not line:
                continue
            kind, outside = line
            elims = [((r, c), {v}) for (r, c) in outside
                     if board.grid[r][c] == 0 and v in board.candidates[r][c]]
            if elims:
                targets = "、".join(_cell(r, c) for (r, c), _ in elims)
                idx = elims[0][0][0] + 1 if kind == "列" else elims[0][0][1] + 1
                reason = (f"第{b + 1}宮中,數字 {v} 的候選位置全部位於第{idx}{kind}"
                          f" → 該{kind}在此宮以外的格子({targets})可刪除候選 {v}")
                return Step("區塊摒除 (Pointing)", "eliminate",
                            [p for p, _ in elims], (v,), reason, 3.5, elims)
    return None


# 技巧施放順序:由簡到難(模擬人類「先找便宜的步」)
TECHNIQUES = [naked_single, hidden_single, naked_pair, pointing]
