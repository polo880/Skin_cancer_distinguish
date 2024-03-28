import tensorflow as tf
import numpy
from tensorflow import keras
from keras.layers import Conv2D, Flatten, BatchNormalization, Dropout, Dense, MaxPool2D
import os

import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.applications import ResNet50V2
from keras.utils.image_utils import img_to_array ,load_img
from keras.utils import to_categorical
from keras.applications.resnet_v2 import preprocess_input
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

normal='Normal skin'
black='暗瘡'
red='紅疹'
skin_folder=['Normal skin','暗瘡','紅疹']
thedir = skin_folder[0]
myna_fnames = os.listdir(thedir)

##取出img並分成data & target
data = []
target = []

for i in range(3):
    thedir =skin_folder[i]
    skin_fnames = os.listdir(thedir)
    for skin in skin_fnames:
        if skin=='.ipynb_checkpoints':
          continue
        img = load_img(thedir + '/' + skin, target_size = (256,256))
        x = img_to_array(img)
        data.append(x)
        target.append(i)
        
data = np.array(data)
n=1;
tmp=np.uint8(data[n])
plt.imshow(tmp);

n=1
#plt.axis('off')
#plt.imshow(data[n]/255);
#plt.show()
x_train = preprocess_input(data)
y_train = to_categorical(target, 3)
X_train,X_valid,Y_train,Y_valid=train_test_split(x_train,y_train,test_size=0.2)

###model##
resnet = ResNet50V2(include_top=False, pooling="avg")
resnet.trainable = False

model = Sequential()
model.add(resnet)
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train,validation_data=(X_valid,Y_valid), batch_size=30, epochs=10)

export_path = 'saved_model'  # 指定路徑
tf.saved_model.save(model, export_path) 

y_predict = np.argmax(model.predict(x_train), -1)
labels = ["正常皮膚", "暗瘡", "紅疹"]