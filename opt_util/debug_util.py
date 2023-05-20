#ライブラリの導入
import numpy as np
import tensorflow as tf

#tensorがtypeで判別できない

def check_arg(x, **kwargs):
    print(type(x))

    if type(x)==np.ndarray:
        return print(type(x),x.shape)


    elif type(x)==tf.EagerTensor:
        return print(type(x),x.shape())




    else :
        return print("numpy,tensor以外です")    
    
    
"""
elif type(x)=='tensorflow.python.framework.ops.EagerTensor':
    return print(type(x),x.shape())
"""
"""
elif x.__class__.__name__ == "tf.Tensor":
    return print(type(x),x.shape())
"""