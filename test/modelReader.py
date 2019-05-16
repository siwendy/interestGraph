'''
Created on 2019年1月4日

@author: zhangyanqing1
'''
import tensorflow as tf

import os
from tensorflow.python import pywrap_tensorflow
# model_dir = r'G:\KeTi\C3D'
# checkpoint_path = os.path.join(model_dir, "sports1m_finetuning_ucf101.model")
checkpoint_path = 'D:/model.ckpt-53000'
# 从checkpoint中读出数据
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# reader = tf.train.NewCheckpointReader(checkpoint_path) #
# 用tf.train中的NewCheckpointReader方法
var_to_shape_map = reader.get_variable_to_shape_map()
# 输出权重tensor名字和值
with open('D:/model_value.txt', 'w') as f:
    for key in var_to_shape_map:
        f.write("tensor_name: " + key + (str)
                (reader.get_tensor(key).shape) + '\n')
        f.write(key + '\n')
        f.write(str(reader.get_tensor(key)) + '\n')
