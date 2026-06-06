# -*- coding: utf-8 -*-
"""
coach.py — 教練模組(系統核心)
理念:不劇透。陪玩家一步一步走,而且第一次用到某招時,順便把「這招怎麼用」教給玩家。
目標不是把數獨解掉,是讓玩家自己變強。
"""
from .board import Board
from .solver import peek_next, apply_step, rate_difficulty


# 每種技巧的「通用心法」——第一次遇到時教給玩家(教練 vs 解題機的差別就在這裡)
TECHNIQUE_LESSONS = {
    "唯一候選數 (Naked Single)":
        "把這格同列、同行、同宮已出現的數字全部排除後,只剩一個可能——那它就一定是它。"
        "最基本的一招,卡住時先掃一遍有沒有這種格子。",
    "隱性唯一數 (Hidden Single)":
        "換個角度想:不是問『這格能填什麼』,而是問『這個數字在這區域還能放哪』。"
        "若某數字在一整列/行/宮裡只剩一格能放,那格就是它——即使那格表面上還有別的候選。",
    "顯性數對 (Naked Pair)":
        "若同一區域裡有兩格的候選數一模一樣、且都只有兩個(例如都是 {3,7}),"
        "這兩個數字就被這兩格鎖住了——同區域其他格都不可能是它們,可以放心刪掉。",
    "區塊摒除 (Pointing)":
        "若某數字在一個宮裡的候選位置剛好全擠在同一列(或行),"
        "那這個數字在那一列、但屬於其他宮的格子就都不可能了,可以刪除。",
}


class CoachSession:
    """一場教練陪玩:逐步給提示、教技巧、追蹤進度,絕不主動劇透整盤答案。"""

    def __init__(self, grid):
        self.board = Board(grid)
        self.taught = set()          # 已經教過的技巧(每招只教一次)
        self.history = []            # 走過的步驟
        self.gave_up = False

    def next_hint(self):
        """回傳一個 dict:這一步的提示 + (若是新招)心法教學。
        回傳 None 代表已解完。回傳 {'stuck': True} 代表基礎技巧用盡。"""
        if self.board.is_solved():
            return None
        step = peek_next(self.board)
        if step is None:
            if self.board.find_contradiction():
                return {"contradiction": True}
            return {"stuck": True}

        lesson = None
        if step.technique not in self.taught:
            lesson = TECHNIQUE_LESSONS.get(step.technique)
            self.taught.add(step.technique)

        return {"step": step, "lesson": lesson,
                "is_new_technique": lesson is not None}

    def accept(self, step):
        """玩家照提示走了一步(或自己填對了)→ 推進盤面。"""
        apply_step(self.board, step)
        self.history.append(step)

    def summary(self):
        label, used = rate_difficulty(self.history)
        return {"difficulty": label, "techniques": used,
                "moves": len(self.history),
                "learned": sorted(self.taught)}
