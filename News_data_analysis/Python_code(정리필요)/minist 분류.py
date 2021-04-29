#mnist 분류해보기
from __future__ import print_function #파이썬 3에서 쓰던 문법을 파이썬 2에서 쓸 수 있게 해주는 문법
import h5py #Python에서 HDF5 바이너리 데이터 포맷을 사용하기 위한 인터페이스 패키지
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np

#반복횟수 와 생성할 cluster 지정
batch_size = 128
num_classes = 10
epochs = 10

#input image dimensions
img_rows, img_cols = 28, 28

#the data 데이터 나누기
(x_train, y_train),(x_test, y_test) = mnist.load_data() #시험용 데이터라 데이터가 쉽게쉽게 나누어 지는 듯

#데이터 확인
plt.imshow(x_train[1000])
plt.show()

if K.image_data_format() == 'channels_first':
    # image_data_format이 ‘channels_first’인 경우 (샘플 수, 채널 수, 행, 열)로 이루어진 4D 텐서
    # image_data_format이 ‘channels_last’인 경우 (샘플 수, 행, 열, 채널 수)로 이루어진 4D 텐서
    # 이미지의 크기를 균일하게 조정
    x_train = x_train.reshape(x_train.shape[0], 1,img_rows,img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1,img_rows,img_cols)
    input_shape = (1, img_rows,img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows,img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32') #

x_train /= 255
x_test /= 255

print('x_train shape:', x_train.shape)

#convert class vector to binary class matrices 벡터를 이진 클래스 매트릭스로 변환
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#학습모델 생성하기
model = Sequential() #
model.add(Conv2D(32, kernel_size=(3,3), input_shape=input_shape))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

#학습절차 구성
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#학습수행
model.fit(x_train, y_train, batch_size=batch_size,epochs=epochs,verbose=1,
          validation_data=(x_test, y_test))

#학습모델 평가
score = model.evaluate(x_test,y_test, verbose=0)
print('test loss',score[0])
print('test accuracy',score[1])


#앙상블 기법 적용해보기
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from keras.layers import BatchNormalization
#여러모델 만들기

model1 = Sequential() #기본 시그모이드에 배치정규화
model1.add(Dense(32, input_shape=input_shape,activation='sigmoid'))
model1.add(BatchNormalization()) #배치정규화
model1.add(Dense(50, activation= 'sigmoid'))
model1.add(BatchNormalization())
model1.add(Dense(50, activation= 'sigmoid'))
model1.add(BatchNormalization())
model1.add(Dense(128, activation= 'sigmoid'))
model1.add(BatchNormalization())
model1.add(Dense(10, activation= 'softmax')) #마지막 출력층은 y범주의 갯수와 맞춰야 하는듯

#학습절차 구성
model1.compile(loss= keras.losses.categorical_crossentropy,
               optimizer=keras.optimizers.Adadelta(),
               metrics=['accuracy'])

#학습 수행
model1.fit(x_train,y_train, batch_size=batch_size,epochs=epochs,verbose=1,
           validation_data=(x_test,y_test))

#학습모델 평가
score1 = model1.evaluate(x_test,y_test, verbose=0)
print('test loss',score1[0])
print('test accuracy',score1[1])




