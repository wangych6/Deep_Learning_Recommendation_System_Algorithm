#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/3/30 15:17
# @author   : chestorwang
# @File    : model.py

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout, Layer
from tensorflow.keras.regularizers import l2

class AttentionLayer(Layer):
    def __init__(self,
                 attention_hidden_units,
                 activation='prelu'):
        super(AttentionLayer, self).__init__()
        self.dense = [Dense(unit_nums, activation=PReLU()) for unit_nums in attention_hidden_units]
        self.out = Dense(1)
    
    def call(self, inputs):
        querry, key, val, mask = inputs
        # 为每一个行为序列的item计算score，方法是先把候选商品展开为和序列一样的维度，也就是maxlen的长度
        querry = tf.tile(querry, multiples=[1, key.shape[1]])
        querry = tf.reshape(querry, shape=[-1, key.shape[1], key.shape[2]])
        # 特征交叉
        outputs = tf.concat([querry, key, querry-key, querry*key], axis=-1)
        # dnn
        for dense in self.dense:
            outputs = dense(outputs)
        # 对于我们padding为0的那些元素，不用参与score计算，所以给他们赋予很小的权重
        outputs = self.out(outputs)
        outputs = tf.squeeze(outputs, axis=-1)
        padding = tf.ones_like(outputs) *(-2 ** 32 + 1)
        outputs = tf.where(tf.equal(mask, 0), padding, outputs)
        # softmax层
        outputs = tf.nn.softmax(logits=outputs)
        outputs = tf.expand_dims(outputs,axis=1)
        # 加权pooling
        outputs = tf.matmul(outputs, val)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class DIN(Model):
    def __init__(self,
                 feature_columns,
                 behavior_feature_list,
                 attention_hidden_units=None,
                 dnn_hidden_units=None,
                 attention_activation='prelu',
                 dnn_activation='relu',
                 dnn_dropout=0,
                 embedding_regularizer=1e-4,
                 sequence_length=50,
                 **kwargs):
        super(DIN, self).__init__()
        if dnn_hidden_units is None:
            dnn_hidden_units = (512, 128, 64)
        if attention_hidden_units is None:
            attention_hidden_units = (64, 32)
        self.sequences_length = sequence_length
        self.dense_feature_info, self.sparse_feature_info = feature_columns

        self.other_sparse_lenght = len(self.sparse_feature_info) - len(behavior_feature_list)
        self.dense_feature_length = len(self.dense_feature_info)
        self.behavior_feature_nums = len(behavior_feature_list)

        self.sparse_features_embedding = [Embedding(input_dim=feat['feat_num'],
                                                    input_length=1,
                                                    output_dim=feat['embed_dim'],
                                                    embeddings_initializer='random_uniform',
                                                    embeddings_regularizer=l2(embedding_regularizer))
                                          for feat in self.sparse_feature_info
                                          if feat['feat'] not in behavior_feature_list]

        self.sequences_features_embedding = [Embedding(input_dim=feat['feat_num'],
                                                       input_length=1,
                                                       output_dim=feat['embed_dim'],
                                                       embeddings_initializer='random_uniform',
                                                       embeddings_regularizer=l2(embedding_regularizer))
                                             for feat in self.sparse_feature_info
                                             if feat['feat'] in behavior_feature_list]

        self.attention_layer = AttentionLayer(attention_hidden_units, activation='prelu')
        self.batchnorm = BatchNormalization(trainable=True)
        self.dnn_layer = [Dense(unit, activation=PReLU()) for unit in dnn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.out = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs, sequense_inputs, item_inputs = inputs
        mask = tf.cast(tf.not_equal(sequense_inputs[:, :, 0], 0), dtype=tf.float32)
        # 非序列的稀疏特征通过embedding后，和稠密特征拼接
        other_inputs = dense_inputs
        for i in range(self.other_sparse_lenght):
            other_inputs = tf.concat([other_inputs, self.sparse_features_embedding[i](sparse_inputs[:, i])], axis=-1)

        # 序列特征enbedding
        sequense_embedding = tf.concat([self.sequences_features_embedding[i](sequense_inputs[:, :, i]) for i in range(self.behavior_feature_nums)], axis=-1)
        item_embedding = tf.concat([self.sequences_features_embedding[i](item_inputs[:, i]) for i in range(self.behavior_feature_nums)], axis=-1)

        # 把序列特征输入attention层
        user_info = self.attention_layer([item_embedding, sequense_embedding, sequense_embedding, mask])
        if self.dense_feature_length > 0 or self.other_sparse_lenght > 0:
            outputs = tf.concat([user_info, item_embedding, other_inputs], axis=-1)
        else:
            outputs = tf.concat([user_info, item_embedding], axis=-1)

        # 送入dnn
        for dense in self.dnn_layer:
            outputs = dense(outputs)

        outputs = self.dropout(outputs)
        outputs = tf.nn.sigmoid(self.out(outputs))

        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_feature_length, ), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_lenght, ), dtype=tf.int32)
        seq_inputs = Input(shape=(self.sequences_length, self.behavior_feature_nums), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_feature_nums, ), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                    outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()


def test_model():
    dense_features = [{'feat': 'a'}, {'feat': 'b'}]
    sparse_features = [{'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'cate_id', 'feat_num': 100, 'embed_dim': 8},
                       {'feat': 'adv_id', 'feat_num': 100, 'embed_dim': 8}]
    behavior_list = ['item_id', 'cate_id']
    features = [dense_features, sparse_features]
    model = DIN(features, behavior_list)
    model.summary()


if __name__ == '__main__':
    test_model()
