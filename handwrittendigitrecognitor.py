import cv2
import os
import numpy as np
import keras
from sklearn.utils import shuffle
from keras.utils import np_utils
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

epochs = 15
batch_size = 70
img_rows = 28
img_cols = 28
#Collecting images for trainig dataset 
train_data_path = "Dataset"
train_data_dir_list = os.listdir(train_data_path)
num_classes = 10
label_names = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}

train_img_data_list = []
train_label_list = []
for ds in train_data_dir_list:
    train_img_list = os.listdir(train_data_path + '/' + ds)
    label = label_names[ds]
    for img in train_img_list:
        input_img = cv2.imread(train_data_path + '/' + ds + '/' + img)
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        train_label_list.append(label)
        train_img_data_list.append(input_img)

train_img_data = np.array(train_img_data_list) 
train_label = np.array(train_label_list)

#Collecting images for test dataset
test_data_path = "TestSample"
test_data_dir_list = os.listdir(test_data_path)
test_img_data_list = []
test_label_list = []
for ds in test_data_dir_list:
    test_img_list = os.listdir(test_data_path + '/' + ds)
    label = label_names[ds]
    for img in test_img_list:
        input_img = cv2.imread(test_data_path + '/' + ds + '/' + img)
        input_img=cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        test_label_list.append(label)
        test_img_data_list.append(input_img)
test_img_data = np.array(test_img_data_list) 
test_label = np.array(test_label_list)
print(test_label)
print(test_img_data.shape)
if K.image_data_format() == 'channels_first':
    train_img_data = train_img_data.reshape(train_img_data.shape[0], 1, img_rows, img_cols)
    test_img_data = test_img_data.reshape(test_img_data.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    train_img_data = train_img_data.reshape(train_img_data.shape[0], img_rows, img_cols, 1)
    test_img_data = test_img_data.reshape(test_img_data.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

train_img_data = train_img_data.astype('float32')
train_img_data /= 255
test_img_data = test_img_data.astype('float32')
test_img_data /= 255
#converting to one hot
Y_train = np_utils.to_categorical(train_label,num_classes)
Y_test = np_utils.to_categorical(test_label,num_classes)

#shuffle the dataset
x_train_data,y_train_data = shuffle(train_img_data,Y_train,random_state = 2)

#model
model = Sequential()

#Convolution layer with relu as activation
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

model.add(Conv2D(64, (3, 3), activation='relu'))

model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

model.fit(train_img_data, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_img_data, Y_test))

score = model.evaluate(test_img_data, Y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
