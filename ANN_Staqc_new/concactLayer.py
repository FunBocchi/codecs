import tensorflow as tf
from tensorflow.keras.layers import Layer

seed = 42
tf.random.set_seed(seed)
random.seed(seed)
np.random.seed(seed)

class ConcatLayer(Layer):
    def __init__(self, **kwargs):
        super(ConcatLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # 将输入张量按照第二个维度进行分割
        block_level_code_output = tf.split(inputs, inputs.shape[1], axis=1)
        # 在第三个维度上进行拼接
        block_level_code_output = tf.concat(block_level_code_output, axis=2)
        # 去除第二个维度的维度
        block_level_code_output = tf.squeeze(block_level_code_output, axis=1)
        print(block_level_code_output)
        return block_level_code_output

    def compute_output_shape(self, input_shape):
        print("===========================", input_shape)
        return (input_shape[0], input_shape[1] * input_shape[2])
