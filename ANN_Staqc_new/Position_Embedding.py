#! -*- coding: utf-8 -*-
# %%
from __future__ import print_function
import tensorflow as tf
import os
import numpy as np
import random
seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):  # 上一层一般就是embedding层，batch_size,seq_len,model_dim
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])  # d_model的长度，比如512
        batch_size, seq_len = tf.shape(x)[0], tf.shape(x)[1]  #
        position_j = 1. / tf.pow(10000., 2 * tf.range(self.size / 2, dtype=tf.float32) / self.size)  #
        position_j = tf.expand_dims(position_j, 0)  # (1,256)
        position_i = tf.cumsum(tf.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = tf.expand_dims(position_i, 2)  # bs,seq_len,1
        position_ij = tf.matmul(position_i, position_j)  # bs,seq_len,256
        position_ij_2i = tf.sin(position_ij)[..., tf.newaxis]  # bs,seq_len,model_dim/2,1
        position_ij_2i_1 = tf.cos(position_ij)[..., tf.newaxis]  # bs,seq_len,model_dim/2,1
        position_ij = tf.concat([position_ij_2i, position_ij_2i_1], axis=-1)  # bs,seq_len,model_dim/2,2
        position_ij = tf.reshape(position_ij, (batch_size, seq_len, self.size))  # bs,seq_len,model_dim

        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return tf.concat([position_ij, x], axis=-1)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)