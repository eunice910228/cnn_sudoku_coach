# -*- coding: utf-8 -*-
"""
grid_extractor.py — 感知模組第一步:從照片中找到數獨盤面
流程:灰階 → 自適應二值化 → 找最大四邊形輪廓 → 透視校正 → 切成 81 格
依賴:opencv-python、numpy(建議在 Colab 執行)
"""
import cv2
import numpy as np

WARP = 450          # 校正後的盤面邊長
CELL = WARP // 9    # 每格 50px


def find_board(image_bgr):
    """回傳透視校正後的 450x450 灰階盤面;找不到回傳 None"""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    quad = _find_quad(contours, image_bgr.shape[0] * image_bgr.shape[1])
    if quad is None:
        return None

    pts = _order_corners(quad.reshape(4, 2).astype(np.float32))
    dst = np.array([[0, 0], [WARP - 1, 0],
                    [WARP - 1, WARP - 1], [0, WARP - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(gray, M, (WARP, WARP))


def _find_quad(contours, img_area):
    """在最大的幾個輪廓中,尋找面積夠大的凸四邊形(盤面外框)

    用遞增的 epsilon 嘗試多邊形逼近:照片中的盤面外框常因格線、
    雜訊或本身已裁切貼邊而呈鋸齒狀,固定 0.02 往往逼近不出 4 個角。
    """
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(c)
        if area < 0.25 * img_area:      # 盤面至少佔畫面的 1/4,否則往下都更小
            break
        peri = cv2.arcLength(c, True)
        for eps in (0.02, 0.03, 0.05, 0.08, 0.10):
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx
    return None


def _order_corners(pts):
    """四個角點排序:左上、右上、右下、左下"""
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype=np.float32)


def slice_cells(board_gray):
    """把盤面切成 81 個 28x28 的格子影像(餵給 CNN 的格式)"""
    cells = []
    for r in range(9):
        for c in range(9):
            cell = board_gray[r * CELL:(r + 1) * CELL,
                              c * CELL:(c + 1) * CELL]
            cell = cell[5:-5, 5:-5]                  # 去掉格線邊緣
            cell = cv2.resize(cell, (28, 28))
            cell = 255 - cell                        # 反相:白字黑底(同訓練資料)
            cells.append(cell.astype("float32") / 255.0)
    return np.array(cells).reshape(-1, 28, 28, 1)
