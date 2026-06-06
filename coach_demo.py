# -*- coding: utf-8 -*-
"""
coach_demo.py — 教練模式展示(零依賴:python3 coach_demo.py)
教練模式:不直接解題,而是陪玩家一步步走,並在第一次用到某招時順便教學。
本檔以「自動走查」模擬一位照著提示前進的玩家,方便展示;
最後附 interactive() 可改成真人互動(input)。
"""
from reasoning.board import Board
from reasoning.coach import CoachSession


def parse(s):
    return [[int(s[r * 9 + c]) for c in range(9)] for r in range(9)]


PUZZLE = parse("300000000970010000600583000200000900500621003008000005000435002000090056000000001")


def walkthrough(grid, show_steps=14):
    coach = CoachSession(grid)
    print("🐱 教練模式開始。我不會直接給你答案,只在你需要時,提示下一步。\n")
    print(Board(grid).pretty())
    print()

    n = 0
    while True:
        hint = coach.next_hint()
        if hint is None:
            print("\n🎉 解開了!——而且是你『在提示下自己走完』的,不是被劇透的。")
            break
        if hint.get("stuck"):
            print("\n🤔 基礎技巧到這裡用盡了。要不要挑戰自己想想?(或啟用回溯法看完整解)")
            break
        if hint.get("contradiction"):
            print("\n⚠️ 盤面出現矛盾,可能是某格輸入錯了(若來自拍照,通常是 CNN 認錯)。")
            break

        step = hint["step"]
        n += 1
        if hint["is_new_technique"]:
            print(f"  🆕 新招解鎖【{step.technique}】")
            print(f"     心法:{hint['lesson']}")
        if n <= show_steps:
            print(f"  第{n:>2}步提示 ▶ {step.reason}")
        elif n == show_steps + 1:
            print("  ……(後續步驟同理,略)")
        coach.accept(step)        # 模擬玩家照提示填了這一步

    s = coach.summary()
    print("\n" + "=" * 60)
    print(f"本局難度:{s['difficulty']}")
    print(f"總步數:{s['moves']}")
    print("這一局你學會的招式:")
    for t in s["learned"]:
        print(f"   ✓ {t}")


def interactive(grid):
    """真人互動版:每次按 Enter 看下一步提示,輸入 q 投降看完整解答。"""
    coach = CoachSession(grid)
    print(Board(grid).pretty())
    while True:
        cmd = input("\n[Enter]=下一步提示  [q]=投降看完整解 > ").strip()
        if cmd == "q":
            from reasoning.solver import solve_with_explanations
            b, steps = solve_with_explanations(grid)
            print("\n(玩家投降,完整解答)")
            print(b.pretty())
            return
        hint = coach.next_hint()
        if hint is None:
            print("🎉 解開了!")
            return
        if hint.get("stuck") or hint.get("contradiction"):
            print("基礎技巧用盡或盤面有誤。")
            return
        step = hint["step"]
        if hint["is_new_technique"]:
            print(f"🆕 新招【{step.technique}】:{hint['lesson']}")
        print(f"提示 ▶ {step.reason}")
        coach.accept(step)


if __name__ == "__main__":
    walkthrough(PUZZLE)
