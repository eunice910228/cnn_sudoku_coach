# -*- coding: utf-8 -*-
"""_labutil.py — learn/ 共用小工具"""


def use_cjk():
    """讓 matplotlib 圖上的中文不會變成豆腐方塊(Windows 與 Colab/Linux 都涵蓋)"""
    import matplotlib
    candidates = [
        "Microsoft JhengHei", "Microsoft YaHei", "SimHei", "PMingLiU",
        "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
        "WenQuanYi Zen Hei", "Taipei Sans TC Beta", "Droid Sans Fallback",
    ]
    existing = matplotlib.rcParams.get("font.sans-serif", [])
    matplotlib.rcParams["font.sans-serif"] = candidates + list(existing)
    matplotlib.rcParams["axes.unicode_minus"] = False
