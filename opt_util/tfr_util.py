#ライブラリの導入
import tensorflow as tf

"""
# 学習データをTFRecordに書き込むためのユーテリティ
def feature_float_list(l):
    return tf.train.Feature(float_list=tf.train.FloatList(value=l))

def record2example(r, length_h, lemgth_w):
    return tf.train.Example(features=tf.train.Features(feature={
        "x": feature_float_list(r[0:length_h*lemgth_w*1]),
        "y": feature_float_list(r[length_h*lemgth_w*1:length_h*lemgth_w*2])
    }))
"""

def record2example(r, length_h, lemgth_w):
    return tf.train.Example(features=tf.train.Features(feature={
        "x": tf.train.Feature(float_list=tf.train.FloatList(value=r[0:length_h*lemgth_w*1])),
        "y": tf.train.Feature(float_list=tf.train.FloatList(value=r[length_h*lemgth_w*1:length_h*lemgth_w*2]))
    }))
