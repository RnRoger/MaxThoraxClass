from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import keras
import numpy as np
import h5py
from generator import XRay_Generator as xg
import os
from fileNamesExtractor import extractImgFileNames as ei
import multiprocessing as mp
import datetime


#Get back the convolutional part of a VGG network trained on ImageNet
model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
model_vgg16_conv.summary()

imgsize = 200
batchsize = 5
no_epoch = 10

files, pre_y = ei.extractImgFileNames("testdata","testdata/overviewTest.csv")

ydict = {}
counter = 0
for yValue in pre_y:
    if yValue not in ydict.keys():
        ydict[yValue] = counter
        counter += 1
y = [ydict[yValue] for yValue in pre_y]

ydict = dict((v,k) for k,v in ydict.items())


y = keras.utils.to_categorical(y,3)

xGen = xg.XRay_Generator(files,y,batchsize,imgsize)

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

i = datetime.datetime.now()
#Then training with your data !
my_model.fit_generator(generator = xGen,
                       steps_per_epoch = (int(len(files)) // batchsize),
                       epochs = no_epoch,
                       verbose = 1,
                       use_multiprocessing = False,
                       workers = mp.cpu_count(),
                       max_queue_size = 8)
print(datetime.datetime.now()-i)
