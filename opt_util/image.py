#ライブラリの導入
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


def input_img(folder_path=".", format="png", mode="normal",file_name="", batch_size=0, num_per_file=0, counter1=0, counter2=0, bias=0, **kwargs):

    if (counter2+1)%4==0:
        print(counter2+1)

    #画像情報のロード、正規化
    x_train=list(range(batch_size))
    if mode=="normal":
        for i in range(batch_size):
            x_train[i] = cv2.imread(folder_path+"/"+str(counter1*num_per_file+counter2*batch_size+i)+"."+format,0)
    elif mode=="inverse":
        for i in range(batch_size):
            x_train[i] = cv2.imread(folder_path+"/"+str(bias-(counter1*num_per_file+counter2*batch_size+i))+"."+format,0)
    elif mode=="single":
        x_train[i] = cv2.imread(folder_path+"/"+file_name+"."+format,0)
    x_train=np.array(x_train)
    x_train=x_train.astype(np.float32)
    
    num = np.shape(x_train)[0]
    height = np.shape(x_train)[1]
    width = np.shape(x_train)[2]

    #物体点情報の作成
    X_train = np.zeros((num, height, width,2), dtype='float32')

    if mode=="normal" or mode=="inverse":
        for i in range(batch_size):
            X_train[i,0:height,0:width,0]=x_train[i,:,:]

    elif mode=="single":
        X_train[i,0:height,0:width,0]=x_train[:,:]
    
    X_train1 = X_train

    return X_train1



def output_img(x, folder_path=".", format="png" , preview = "on", file_name="count", **kwargs):

    num = np.shape(x)[0]
    height = np.shape(x)[1]
    width = np.shape(x)[2]

    if preview == "on":
        
        col = 3 # 列数
        row = int(num/3+1) # 行数
        fig, ax = plt.subplots(nrows=row,ncols=col,  figsize=(20,10*row))
        #一枚に非対応
        for i in range(num):
            _r= i//col
            _c= i%col
            ax[_r,_c].set_title(i, fontsize=16, color='black')
            ax[_r,_c].imshow(x[i],cmap="gray") # 画像を表示
    else :
        pass

    x2 = np.zeros((num,height,width))

    #256階調化
    for i in range(num):
        min = np.min(x[i])
        max = np.max(x[i])
        x2[i] = 255*((x[i]-min)/(max-min))
    

    #グレースケールで書き出し
    if file_name=="count":
        for i in range(num):
            x3 = x2[i]
            cv2.imwrite(folder_path+"/"+str(i)+"."+format, x3)
    
    else:
        for i in range(num):
            x3 = x2[i]
            cv2.imwrite(folder_path+"/"+file_name+"."+format, x3)

