from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
import tensorflow as tf
import os
import numpy as np
import random

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 检查输入形状是否为列表且长度为2
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('注意力层的输入应该是一个包含2个输入的列表。')
        
        # 检查嵌入维度是否相同
        if not input_shape[0][2] == input_shape[1][2]:
            raise ValueError('嵌入维度应该相同。')

        # 创建可训练的权重矩阵
        self.kernel = self.add_weight(shape=(input_shape[0][2], input_shape[0][2]),         
                                      initializer='glorot_uniform',
                                      name='kernel',
                                      trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # 计算注意力权重
        a = K.dot(inputs[0], self.kernel)
        y_trans = K.permute_dimensions(inputs[1], (0, 2, 1))
        b = K.batch_dot(a, y_trans, axes=[2, 1])
        
        # 应用激活函数
        return K.tanh(b)
    
    def compute_output_shape(self, input_shape):
        # 输出形状与第一个输入的形状相同
        return (None, input_shape[0][1], input_shape[1][1])
