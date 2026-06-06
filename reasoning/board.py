# -*- coding: utf-8 -*-
"""
board.py — 盤面模組
負責:盤面狀態、候選數(candidates)的維護。
這是推理引擎的「世界模型」:每個空格目前還可能填哪些數字。
"""


class Board:
    def __init__(self, grid):
        """grid: 9x9 的整數列表,0 代表空格"""
        self.grid = [row[:] for row in grid]
        self.candidates = [[set() for _ in range(9)] for _ in range(9)]
        self._init_candidates()

    # ---------- 候選數初始化與維護 ----------

    def _peers_values(self, r, c):
        """取得 (r,c) 同列、同行、同宮已出現的數字"""
        used = set(self.grid[r]) | {self.grid[i][c] for i in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        used |= {self.grid[i][j] for i in range(br, br + 3) for j in range(bc, bc + 3)}
        used.discard(0)
        return used

    def _init_candidates(self):
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0:
                    self.candidates[r][c] = set(range(1, 10)) - self._peers_values(r, c)

    def place(self, r, c, v):
        """在 (r,c) 填入 v,並把 v 從所有同列/行/宮的候選數中移除"""
        self.grid[r][c] = v
        self.candidates[r][c] = set()
        for j in range(9):
            self.candidates[r][j].discard(v)
        for i in range(9):
            self.candidates[i][c].discard(v)
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                self.candidates[i][j].discard(v)

    # ---------- 單位(unit)迭代:9 列、9 行、9 宮 ----------

    def units(self):
        for i in range(9):
            yield f"第{i + 1}列", [(i, j) for j in range(9)]
        for j in range(9):
            yield f"第{j + 1}行", [(i, j) for i in range(9)]
        for b in range(9):
            br, bc = 3 * (b // 3), 3 * (b % 3)
            yield f"第{b + 1}宮", [(br + i, bc + j) for i in range(3) for j in range(3)]

    # ---------- 狀態查詢 ----------

    def is_solved(self):
        return all(self.grid[r][c] != 0 for r in range(9) for c in range(9))

    def find_contradiction(self):
        """若有空格已無任何候選數,代表盤面矛盾(常見原因:CNN 辨識錯誤)。
        這就是『推理模組替感知模組把關』的整合點。"""
        for r in range(9):
            for c in range(9):
                if self.grid[r][c] == 0 and not self.candidates[r][c]:
                    return (r, c)
        return None

    def pretty(self):
        lines = []
        for r in range(9):
            if r % 3 == 0 and r:
                lines.append("------+-------+------")
            row = []
            for c in range(9):
                if c % 3 == 0 and c:
                    row.append("|")
                row.append(str(self.grid[r][c]) if self.grid[r][c] else ".")
            lines.append(" ".join(row))
        return "\n".join(lines)
