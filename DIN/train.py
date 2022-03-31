#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 23:00
# @author   : chestorwang
# @File    : train.py.py
import os
import sys
from time import time
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

from model import DIN
from features_processor.UserBehavior import user_behavior

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train():
    attention_hidden_units = [64, 32]
    dnn_hidden_units = [512, 128, 64]
    attention_activation = 'prelu'
    dnn_activation = 'relu'
    dnn_dropout = 0.5
    sequence_length = 50

    learning_rate = 0.001
    batch_size = 1024
    epochs = 20
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = user_behavior()
    print('train', feature_columns, behavior_list, train)
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test
    # ============================Build Model==========================
    model = DIN(feature_columns=feature_columns,
                behavior_feature_list=behavior_list,
                attention_hidden_units=attention_hidden_units,
                dnn_hidden_units=dnn_hidden_units,
                attention_activation=attention_activation,
                dnn_activation=dnn_activation,
                dnn_dropout=dnn_dropout,
                sequence_length=sequence_length)

    model.summary()
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(val_X, val_y),
        batch_size=batch_size,
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y, batch_size=batch_size)[1])


train()
