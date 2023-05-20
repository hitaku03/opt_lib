#ライブラリの導入
import tensorflow as tf
import numpy as np

def convex_lens(x, wl, p, f1 , limit = "on", **kwargs):

    #パラメータの用意

    num = np.shape(x)[0]
    nx = np.shape(x)[2]
    ny = np.shape(x)[1]
    ch = np.shape(x)[3]

    #複素数化
    f=tf.complex(x[:,:,:,0],x[:,:,:,1])

    #位相変調パターンの作成
    lim1 = int((wl*f1)/(2*p*p))
    if lim1 <= (ny/2) and limit == "on":
        x1, y1 = tf.meshgrid(tf.linspace(-lim1, lim1, tf.cast(lim1*2, tf.int32)),tf.linspace(-lim1, lim1, tf.cast(lim1*2, tf.int32)))
    else : 
        x1, y1 = tf.meshgrid(tf.linspace(-ny/2, ny/2, tf.cast(ny, tf.int32)),tf.linspace(-nx/2, nx/2, tf.cast(nx, tf.int32)))
    fx1 = tf.cast(x1,tf.float64)*p
    fy1 = tf.cast(y1,tf.float64)*p
    ph1 = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),tf.dtypes.cast(-1*(np.pi*(pow(fx1,2)+pow(fy1,2)))/(wl*f1),tf.float64)))
    ph1 = tf.cast(ph1, tf.complex64)
    if lim1 <= (ny/2) and limit == "on":
        ph1 = tf.pad(ph1,[[ny/2-lim1,ny/2-lim1],[nx/2-lim1,nx/2-lim1]])
    
    #位相変調の実行
    f = tf.math.multiply(f, ph1)
    f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)

    return f


def double_convex_lens(x, wl, p, f1 , f2, limit = "on", **kwargs):

    #パラメータの用意
    num = np.shape(x)[0]
    nx = np.shape(x)[2]
    ny = np.shape(x)[1]
    ch = np.shape(x)[3]


    #二重焦点用
    tile1 = np.zeros((2,2))
    tile1[0,1] = 1
    tile1[1,0] = 1
    tile1 = np.tile(tile1,(int(ny/2),int(nx/2)))
    tile2 = np.zeros((2,2))
    tile2[1,1] = 1
    tile2[0,0] = 1
    tile2 = np.tile(tile2,(int(ny/2),int(nx/2)))

    #複素数化
    f=tf.complex(x[:,:,:,0],x[:,:,:,1])

    #1つ目の焦点距離
    lim1 = int((wl*f1)/(2*p*p))
    if lim1 <= (ny/2) and limit == "on":
        x1, y1 = tf.meshgrid(tf.linspace(-lim1, lim1, tf.cast(lim1*2, tf.int32)),tf.linspace(-lim1, lim1, tf.cast(lim1*2, tf.int32)))
    else : 
        x1, y1 = tf.meshgrid(tf.linspace(-ny/2, ny/2, tf.cast(ny, tf.int32)),tf.linspace(-nx/2, nx/2, tf.cast(nx, tf.int32)))
    fx1 = tf.cast(x1,tf.float64)*p
    fy1 = tf.cast(y1,tf.float64)*p
    ph1 = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),tf.dtypes.cast(-1*(np.pi*(pow(fx1,2)+pow(fy1,2)))/(wl*f1),tf.float64)))
    ph1 = tf.cast(ph1, tf.complex64)
    if lim1 <= (ny/2) and limit == "on":
        ph1 = tf.pad(ph1,[[ny/2-lim1,ny/2-lim1],[nx/2-lim1,nx/2-lim1]])

        
    #2つ目の焦点距離
    lim2 = int((wl*f2)/(2*p*p))
    if lim2 <= (ny/2) and limit == "on":
        x2, y2 = tf.meshgrid(tf.linspace(-lim2, lim2, tf.cast(lim2*2, tf.int32)),tf.linspace(-lim2, lim2, tf.cast(lim2*2, tf.int32)))
    else :
        x2, y2 = tf.meshgrid(tf.linspace(-ny/2, ny/2, tf.cast(ny, tf.int32)),tf.linspace(-nx/2, nx/2, tf.cast(nx, tf.int32)))
    fx2 = tf.cast(x2,tf.float64)*p
    fy2 = tf.cast(y2,tf.float64)*p
    ph2 = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),tf.dtypes.cast(-1*(np.pi*(pow(fx2,2)+pow(fy2,2)))/(wl*f2),tf.float64)))
    ph2 = tf.cast(ph2, tf.complex64)
    if lim2 <= (ny/2) and limit == "on":
        ph2 = tf.pad(ph2,[[ny/2-lim2,ny/2-lim2],[nx/2-lim2,nx/2-lim2]])

    #2重焦点レンズ用に合成
    ph3 = tf.math.multiply(tile1, ph1)+tf.math.multiply(tile2, ph2)

    #位相変調の実行
    f = tf.math.multiply(f, ph3)
    f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)

    return f



"""
class lens(tf.keras.layers.Layer):

    def __init__(self, output_dim, wl, p, z1 , z2, mode, **kwargs):
        super(lens, self).__init__(**kwargs)        
        wl = wl
        p = p
        z1 = z1
        z2 = z2
        mode = mode
        output_dim = output_dim

    def build(self, input_shape):
        super(lens, self).build(input_shape)
        shape = input_shape
        nx = input_shape[2]
        ny = input_shape[1]
        ch = input_shape[3]
        nx2 = nx * 2
        ny2 = ny * 2
        px =  1 / (nx2 * p)
        py =  1 / (ny2 * p)

    def call(self, x):

        #新規作成
        tile1 = np.zeros((2,2))
        tile1[0,1] = 1
        tile1[1,0] = 1
        tile1 = np.tile(tile1,(int(ny/2),int(nx/2)))

        tile2 = np.zeros((2,2))
        tile2[1,1] = 1
        tile2[0,0] = 1
        tile2 = np.tile(tile2,(int(ny/2),int(nx/2)))
        #新規作成終了

        if mode=="yes":
            f=tf.complex(x[:,:,:,0],x[:,:,:,1])

            #1つ目の焦点距離
            lim1 = int((wl*z1)/(2*p*p))
            if lim1 >(ny/2):
                x1, y1 = tf.meshgrid(tf.linspace(-ny/2, ny/2, tf.cast(ny, tf.int32)),tf.linspace(-nx/2, nx/2, tf.cast(nx, tf.int32)))
            else : 
                x1, y1 = tf.meshgrid(tf.linspace(-lim1, lim1, tf.cast(lim1*2, tf.int32)),tf.linspace(-lim1, lim1, tf.cast(lim1*2, tf.int32)))
            fx1 = tf.cast(x1,tf.float64)*p
            fy1 = tf.cast(y1,tf.float64)*p
            ph1 = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),tf.dtypes.cast(-1*(np.pi*(pow(fx1,2)+pow(fy1,2)))/(wl*z1),tf.float64)))
            ph1 = tf.cast(ph1, tf.complex64)
            if lim1 > (ny/2):
                pass
            else :
                ph1 = tf.pad(ph1,[[ny/2-lim1,ny/2-lim1],[nx/2-lim1,nx/2-lim1]])

            #2つ目の焦点距離
            lim2 = int((wl*z2)/(2*p*p))
            if lim2 > (ny/2):
                x2, y2 = tf.meshgrid(tf.linspace(-ny/2, ny/2, tf.cast(ny, tf.int32)),tf.linspace(-nx/2, nx/2, tf.cast(nx, tf.int32)))
            else :
                x2, y2 = tf.meshgrid(tf.linspace(-lim2, lim2, tf.cast(lim2*2, tf.int32)),tf.linspace(-lim2, lim2, tf.cast(lim2*2, tf.int32)))
            fx2 = tf.cast(x2,tf.float64)*p
            fy2 = tf.cast(y2,tf.float64)*p
            ph2 = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),tf.dtypes.cast(-1*(np.pi*(pow(fx2,2)+pow(fy2,2)))/(wl*z2),tf.float64)))
            ph2 = tf.cast(ph2, tf.complex64)
            if lim2 > (ny/2):
                pass
            else :
                ph2 = tf.pad(ph2,[[ny/2-lim2,ny/2-lim2],[nx/2-lim2,nx/2-lim2]])
            

            #合成
            ph3 = tf.math.multiply(tile1, ph1)+tf.math.multiply(tile2, ph2)

            #掛け算
            f = tf.math.multiply(f, ph3)
            f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)
            return f
        
        elif mode=="no":
            f = tf.complex(x[:,:,:,0],x[:,:,:,1])
            x, y = tf.meshgrid(tf.linspace(-ny/2, ny/2, tf.cast(ny, tf.int32)),tf.linspace(-nx/2, nx/2, tf.cast(nx, tf.int32)))
            fx = tf.cast(x,tf.float64)*p
            fy = tf.cast(y,tf.float64)*p
            ph = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),tf.dtypes.cast(-1*(np.pi*(pow(fx,2)+pow(fy,2)))/(wl*z1),tf.float64)))
            ph = tf.cast(ph, tf.complex64)
            f = tf.math.multiply(f, ph)
            f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)
            return f
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], output_dim)

"""