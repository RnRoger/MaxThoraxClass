# Thorax abnormality classifier

## Usage
Our classfier is made using the NIH x-ray pictures of patiens which have certain abnormalities in their thorax.
We extracted the pictures related to atelectasis, infiltrations and nodules in patients who are twenty years or older using the `fileExtractor`
module.The pictures only contains 1 of these abnormalities at the same time.
Our classfier is capable and can be used to detect atelectasis, infiltrations and nodules in x-ray pictues of the thorax.

## Classifier

Our classifier uses a convolutional neural network(cnn) from keras based on the vgg16 model. it can receive images as input , in our case thorax 
x- ray pictures, and converts it to a image with the size of 255x255. The images are downsized to remove complication with the memory. To further
decrease complications with the memory the XRay_Generator script was written to train the cnn with a few pictures at a time , the amount of pictures
is determined by the batchsize. The true label of the pictures are fetched using the fileNamesExtractor module so it can be used to train the cnn.
The x-ray pictures and the true labels are used to train the cnn. The optimizer algoritm used  for our cnn is  Adam optimization algorithm.
The loss function that was used is called categorical_crossentropy. To later evalutate the performance of our classifier we created a
test set which consisted of 20% of the available data. The remaining 80% of the train data was further divided to create the validation set which consisted
of 20% of the current training data. The validation set is used during the training of the model to optimize its parameters.When the model is done
training its performance is evaluated using the test set. To validate the performence we created a confusion matrix and calculated the accuracy.
The modeles that are created by our algoritm are stored using the save_weights function


