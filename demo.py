# -*- coding: utf-8 -*-
"""
demo.py — 推理引擎展示(零依賴:python3 demo.py 直接執行)
展示四件事:教練提示、思路重播、難度評級、矛盾偵測(感知把關)。
"""
from reasoning.board import Board
from reasoning.solver import solve_with_explanations, hint, rate_difficulty


def parse(s):
    return [[int(s[r * 9 + c]) for c in range(9)] for r in range(9)]


MAIN = parse("300000000970010000600583000200000900500621003008000005000435002000090056000000001")
EASY = parse("530070000600195000098000060800060003400803001700020006060000280000419005000080079")
HARD = parse("043080250600000000000001094900004070000608000010200003820500000000000005034090710")

print("=" * 64)
print("【原始盤面】(中等難度範例)")
print(Board(MAIN).pretty())

print("\n" + "=" * 64)
print("【教練模式】玩家卡住 → 只給『下一步』,不劇透:")
s = hint(MAIN)
print(f"  ▶ 技巧:{s.technique}")
print(f"  ▶ 提示:{s.reason}")

print("\n" + "=" * 64)
print("【思路重播】完整推理鏈(每一步都有理由):")
board, steps = solve_with_explanations(MAIN)
for i, s in enumerate(steps, 1):
    print(f"  {i:>2}. [{s.technique}] {s.reason}")

print("\n" + "=" * 64)
print("【解完盤面】")
print(board.pretty())

label, used = rate_difficulty(steps)
print("\n【難度評級】", label)
print("【技巧統計】", "、".join(f"{t} ×{n}" for t, n in used.items()))

print("\n" + "=" * 64)
print("【難度量化展示】同一引擎,三種題目:")
for name, g in [("簡單題", EASY), ("中等題", MAIN), ("困難題", HARD)]:
    _, st = solve_with_explanations(g)
    lb, _ = rate_difficulty(st)
    print(f"  {name}:{lb}")

print("\n" + "=" * 64)
print("【矛盾偵測】模擬 CNN 把 R1C2 認錯成 7:")
bad = [row[:] for row in MAIN]
bad[0][1] = 7
_, bs = solve_with_explanations(bad)
print(f"  ▶ {bs[-1].reason}")
print("  ▶ 整合點:推理模組(白盒)反過來替感知模組(黑盒)把關。")
