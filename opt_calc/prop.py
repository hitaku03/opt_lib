#ライブラリの導入
import tensorflow as tf
import numpy as np
import math

def angular_spectrum(x, wl, p, z, limit="on", out_type="tensor", out_form="chanel_float", **kwargs):

    #変数を用意
    num = np.shape(x)[0]
    nx = np.shape(x)[2]
    ny = np.shape(x)[1]
    ch = np.shape(x)[3]

    nx2 = nx * 2
    ny2 = ny * 2
    px =  1 / (nx2 * p)
    py =  1 / (ny2 * p)

    #元画像をフーリエ変換
    f=tf.complex(x[:,:,:,0],x[:,:,:,1])
    f=tf.pad(f,[[0,0],[ny//2,ny//2],[nx//2,nx//2]])
    f=tf.signal.fft2d(tf.cast(f, tf.complex64))
    
    #伝達関数の計算をする
    lim = int(((nx2*p)**2)/(wl*math.sqrt(4*z**2+(nx2*p)**2)))
    if lim <= ny and limit == "on":
        x, y = tf.meshgrid(tf.linspace(-lim, lim, tf.cast(lim*2, tf.int32)),tf.linspace(-lim, lim, tf.cast(lim*2, tf.int32)))
    else :
        x, y = tf.meshgrid(tf.linspace(-ny2/2, ny2/2, tf.cast(ny2, tf.int32)),tf.linspace(-nx2/2, nx2/2, tf.cast(nx2, tf.int32)))
    fx = tf.cast(x,tf.float64)*px
    fy = tf.cast(y,tf.float64)*py
    ph = tf.exp(tf.dtypes.complex(tf.dtypes.cast(0.,tf.float64),+2*np.pi*z*tf.sqrt(1/(wl*wl)-(fx**2+fy**2))))
    if lim <= ny and limit == "on":
        ph = tf.pad(ph,[[ny-lim,ny-lim],[nx-lim,nx-lim]])
    ph = tf.cast(ph, tf.complex64)
    ph = tf.signal.fftshift(ph)

    #角スペクトル法の実行
    f = tf.math.multiply(f, ph)
    f = tf.signal.ifft2d(tf.cast(f, tf.complex64))
    f = tf.slice(f, (0, ny//2, nx//2), (-1, ny, nx))
    
    #出力形式の選択
    if out_form == "complex":
        if out_type == "numpy":
            f = f.numpy()
            return f
    
        elif out_type == "tensor":
            return f

    elif out_form == "chanel_float":
        f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)
        if out_type == "numpy":
            f = f.numpy()
            return f
    
        elif out_type == "tensor":
            return f
    
    elif out_form == "amp_float":
        f = tf.concat([tf.expand_dims(tf.math.real(f), axis=-1), tf.expand_dims(tf.math.imag(f), axis=-1)], 3)
        if out_type == "numpy":
            hol_pred_32_amp = tf.abs(tf.complex(f[:,:,:,0],f[:,:,:,1]))
            hol = hol_pred_32_amp.numpy()
            return hol
    
        elif out_type == "tensor":
            hol_pred_32_amp = tf.abs(tf.complex(f[:,:,:,0],f[:,:,:,1]))
            return hol_pred_32_amp