from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense
from keras.optimizers import Adam
from keras.models import Model
import keras
import tensorflow as tf
from generator import XRay_Generator as xg
from fileNamesExtractor import extractImgFileNames as ei
import multiprocessing as mp
import datetime
import os

def getGen(files,pre_y, batchsize, imgsize):
    ydict = {}
    counter = 0
    for yValue in pre_y:
        if yValue not in ydict.keys():
            ydict[yValue] = counter
            counter += 1
    y = [ydict[yValue] for yValue in pre_y]
    y = keras.utils.to_categorical(y, 3)

    xGen = xg.XRay_Generator(files, y, batchsize, imgsize)

    return xGen, y

if __name__ == "__main__":
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    #Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    model_vgg16_conv.summary()

    imgsize = 200
    batchsize = 5
    no_epoch = 10
    use_multiprocessing = False

    trainFiles, trainPre_y, validateFiles, validatePre_y, testFiles, testPre_y = ei.extractImgFileNames("testdata","testdata/overviewTest.csv")

    trainGen, trainY = getGen(trainFiles,trainPre_y, batchsize, imgsize)
    validateGen, validateY = getGen(validateFiles, validatePre_y, batchsize, imgsize)
    testGen, testY = getGen(testFiles, testPre_y, batchsize, imgsize)

    input = Input(shape=(imgsize,imgsize,3),name = 'image_input')


    #Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    #Add the fully-connected layers
    x1 = Flatten(name='flatten')(output_vgg16_conv)
    x1 = Dense(200, activation='relu', name='fc1')(x1)
    x1 = Dense(200, activation='relu', name='fc2')(x1)
    x1 = Dense(3, activation='softmax', name='predictions')(x1)

    #Create your own model
    my_model = Model(input=input, output=x1)

    Adam = Adam(lr=.0001)
    my_model.compile(optimizer=Adam, loss ='categorical_crossentropy', metrics=['accuracy'])
    #In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
    my_model.summary()

    i = datetime.datetime.now()
    #Then training with your data !
    my_model.fit_generator(generator = trainGen,
                           steps_per_epoch = (int(len(trainFiles)) // batchsize),
                           epochs = no_epoch,
                           verbose = 1,
                           use_multiprocessing = use_multiprocessing,
                           validation_data= validateGen,
                           validation_steps= (int(len(validateFiles)) // batchsize),
                           workers = mp.cpu_count(),
                           max_queue_size = 8)
    print(datetime.datetime.now()-i)

    score = my_model.evaluate_generator(testGen, len(testFiles) / batchsize, workers= mp.cpu_count())
    print("Loss: ", score[0], "Accuracy: ", score[1])

    model_json = my_model.to_json()
    with open("models"+os.sep()+datetime.datetime.today().strftime('%Y-%m-%d')+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    my_model.save_weights("model.h5")
    print("Saved model to disk")
