# -*- coding: utf-8 -*-
"""
digit_cnn.py — 感知模組的核心:小型 CNN(LeNet 等級)
輸入 28x28 灰階格子影像 → 輸出 10 類(0=空格、1~9=數字)。
Conv 層學到的是一組濾波器;第一層 filter 可視覺化觀察其學到的特徵。
依賴:tensorflow / keras(Colab 內建)
"""
from tensorflow import keras
from tensorflow.keras import layers


def build_model():
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation="relu"),   # 32 個 3x3 濾波器:學「筆畫偵測器」
        layers.MaxPooling2D(),                     # 降採樣:獲得平移容忍度
        layers.Conv2D(64, 3, activation="relu"),   # 更深層:組合出「數字部件」
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),                       # 防過擬合
        layers.Dense(10, activation="softmax"),    # 10 類機率輸出
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
