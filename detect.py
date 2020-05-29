import pandas as pd
import numpy as np
import librosa

import librosa
def predictau(filename):
    file_name=filename
    audio, sr = librosa.load(file_name)
    sec=int(len(audio)/sr)
    sec=int(sec/2)
    t=[]
    for i in range(sec):
        y,sr=librosa.load(file_name,offset=2*i,duration=2)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        #print(2*i)
        t.append(ps)

    for i in range(len(t)):
        t[i]=np.reshape(t[i],(1,128,87,1))

    import tensorflow as tf
    model=tf.keras.models.load_model('weights11.49-0.82.hdf5')

    y=[]
    for i in range(len(t)):
        y.append(model.predict(t[i]).tolist())
    out=[]
    for i in range(len(y)):
        if y[i][0][1]>0.473:
            out.append(1)
        elif y[i][0][2]>0.71:
            out.append(2)
        else:
            out.append(0)
    return out

def predictac(filename):
    import tensorflow as tf
    model1=tf.keras.models.load_model('cough.h5')
    import cv2
    mport imutils

    cap = cv2.VideoCapture(filename)
    count = 0
    a=[]
    total=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print(total)
    fps=cap.get(cv2.CAP_PROP_FPS)
    print(fps)
    x=fps*2/20
    while cap.isOpened():
        ret, frame = cap.read()
        #frame=cv2.resize(frame,(224,224))
        if ret:
            a.append(frame)
            count += x# i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break
    data=[]
    if len(a)>(int(len(a)/20))*20:
        a=a[:(int(len(a)/20))*20]
    for i in range(len(a)):
        a[i]=cv2.resize(a[i],(224,224))
        a[i]=a[i]/225
    for i in range(int(len(a)/20)):
        data.append(np.stack(a[i*20:(i+1)*20],axis=0))
    for i in range(len(data)):
        data[i]=np.reshape(data[i],(1,20,224,224,3))
    out=[]
    for i in range(len(data)):
        out.append(model1.predict_classes(data[i]))
    return out
