from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model
import keras
import tensorflow as tf
from generator import XRay_Generator as xg
from fileNamesExtractor import extractImgFileNames as eifn
import multiprocessing as mp
import datetime
import os

# Function to build the classifier model
def buildModel(xrayPath = r"D:\ChestXray-NIHCC\dataset",
               overviewPath = r"D:\ChestXray-NIHCC\overview.csv",
               imgsize = 255,
               batchsize = 6,
               no_epoch = 1,
               use_multiprocessing = False):
    gpu_options = tf.GPUOptions(allow_growth=True)
    session = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

    # Get back the convolutional part of a VGG network trained on ImageNet
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
    model_vgg16_conv.summary() # prints the layers inside the vgg16

    trainFiles, trainPre_y, validateFiles, \
    validatePre_y, testFiles, testPre_y = eifn.extractImgFileNames(xrayPath,
                                                                 overviewPath)

    trainGen, trainY = getGen(trainFiles, trainPre_y, batchsize, imgsize)
    validateGen, validateY = getGen(validateFiles, validatePre_y, batchsize, imgsize)
    testGen, testY = getGen(testFiles, testPre_y, batchsize, imgsize)

    input = Input(shape=(imgsize, imgsize, 3), name='image_input')

    # Use the generated model
    output_vgg16_conv = model_vgg16_conv(input)

    # Add the fully-connected layers
    x1 = Flatten(name='flatten')(output_vgg16_conv)
    x1 = Dense(200, activation='relu', name='fc1')(x1)
    x1 = Dropout(0.1)(x1)
    x1 = Dense(200, activation='relu', name='fc2')(x1)
    x1 = Dropout(0.1)(x1)
    x1 = Dense(3, activation='softmax', name='predictions')(x1)

    #Create your own model
    model = Model(input=input, output=x1)

    adam = Adam(lr = 0.01)
    model.compile(optimizer=adam, loss ='categorical_crossentropy', metrics=['accuracy'])
    model.summary() # print the layers used in the model

    i = datetime.datetime.now()
    #Then training with your data !
    model.fit_generator(generator = trainGen,
                        steps_per_epoch = (int(len(trainFiles)) // batchsize),
                        epochs = no_epoch,
                        verbose = 1,
                        use_multiprocessing = use_multiprocessing,
                        validation_data= validateGen,
                        validation_steps= (int(len(validateFiles)) // batchsize),
                        workers = mp.cpu_count(),
                        max_queue_size = 1)

    print("\nFinished making model.\nExecution time: "+str(datetime.datetime.now()-i))

    # Determine the accuracy of the model with the test data
    score = model.evaluate_generator(testGen, len(testFiles) / batchsize, workers= mp.cpu_count())
    print("Results:\nLoss: ", score[0], "Accuracy: ", score[1])

    # Save the model
    model.save("models"+os.sep+str(datetime.datetime.today().strftime('%Y-%m-%d'))+".h5")

    print("Saved model in: "+os.getcwd()+os.sep+"models")
    print("You can load your model in an other session using:\n"
          "from keras.model import load_model\n"
          "model = load_model(\"PATH TO MODEL.h5\")")

    # Return model and used sets.
    return model, \
           [trainFiles,trainY,trainGen], \
           [validateFiles,validateY,validateGen] ,\
           [testFiles, testY, testGen]

# Function to create a generator and translate pre Y values to usable values
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