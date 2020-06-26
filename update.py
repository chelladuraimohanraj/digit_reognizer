import os
from keras import backend as K
import cv2
from google.colab.patches import cv2_imshow
import keras
import numpy as np
path='d'
mylist=os.listdir(path)
noofclasses=len(mylist)
print('total class=',noofclasses)
classno=[]
images=[]
print('importing..')
for x in range(0,noofclasses):
    mypiclist=os.listdir(path+"/"+str(x))
    for y in mypiclist:
        curimg=cv2.imread(path+"/"+str(x)+"/"+y)
        curimg=cv2.resize(curimg,(28,28))
        images.append(curimg)
        classno.append(x)
    print(x,end=" ")
cv2_imshow(images[0])
images=np.array(images)
classno=np.array(classno)
print('')
print(images.shape)
print(classno.shape)
img_rows, img_cols = 28, 28
def pre(img):
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,img=cv2.threshold(img,50,150,cv2.THRESH_BINARY_INV)
    img=cv2.equalizeHist(img)
    im=img/255
    return img
x_train=np.array(list(map(pre,images)))
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
y_train = keras.utils.to_categorical(classno, noofclasses)

model=keras.models.load_model("model.h5")


batch_size = 128
num_classes = 10
epochs = 50

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          )


model.save("model.h5")