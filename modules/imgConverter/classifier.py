from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import keras
import numpy as np
import h5py
import imgConverter as ic

#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

imgsize = 200

x, ids, pre_y = ic.imgConverter("data",imgsize)

ydict = {}
counter = 0
for yValue in pre_y:
    if yValue not in ydict.keys():
        ydict[yValue] = counter
        counter += 1
y = [ydict[yValue] for yValue in pre_y]

ydict = dict((v,k) for k,v in ydict.items())


y = keras.utils.to_categorical(y,3)


input = Input(shape=(imgsize,imgsize,3),name = 'image_input')


#Use the generated model
output_vgg16_conv = model_vgg16_conv(input)

#Add the fully-connected layers
x1 = Flatten(name='flatten')(output_vgg16_conv)
x1 = Dense(4096, activation='relu', name='fc1')(x1)
x1 = Dense(4096, activation='relu', name='fc2')(x1)
x1 = Dense(3, activation='softmax', name='predictions')(x1)

#Create your own model
my_model = Model(input=input, output=x1)

Adam = Adam(lr=.0001)
my_model.compile(optimizer=Adam, loss ='categorical_crossentropy', metrics=['accuracy'])
#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()


#Then training with your data !
my_model.fit(x = x, y=y, epochs=10,verbose=1)
