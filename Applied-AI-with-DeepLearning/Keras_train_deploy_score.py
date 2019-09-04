#!/usr/bin/env python
# coding: utf-8

# # Zero to Singularity: Create, Tune, Deploy and Scale a Deep Neural Network in 90 Minutes
# 
# This notebook is part of a masterclass held at IBM Think on 13th of February 2019 in San Fransisco
# In this exercise you will train a Keras DeepLearning model running on top of TensorFlow. 
# 
# Note: For sake of bringing the training runtime down we've done two things
# 
# 1) Used a softmax regression model over a Convolutional Neural Network 
# 
# 2) Trained only for one epoch instead of 20
# 
# This leads to approx. 5% less accuracy
# 
# 
# Authors
# 
# Romeo Kienzler - Chief Data Scientist, IBM Watson IoT
# 
# Krishnamurthy Arthanarisamy - Architect, Watson Machine Learning Software Lab, Bangalore
# 
# 
# # Prerequisites
# 
# Please make sure the currently installed version of Keras and Tensorflow are matching the requirememts, if not, please run the two PIP commands below in order to re-install. Please restart the kernal before proceeding, please re-check if the versions are matching.

# In[ ]:


import keras
print('Current:\t', keras.__version__)
print('Expected:\t 2.1.3')


# In[ ]:


import tensorflow as tf
print('Current:\t', tf.__version__)
print('Expected:\t 1.5.0')


# # IMPORTANT !!!
# 
# If you ran the two lines below please restart your kernel (Kernel->Restart & Clear Output)

# In[ ]:


get_ipython().system('pip install keras==2.1.3')


# In[ ]:


get_ipython().system('pip install tensorflow==1.5.0')


# # 1.0 Train a MNIST digits recognition model
# We start with some global parameters and imports

# In[ ]:


#some learners constantly reported 502 errors in Watson Studio. 
#This is due to the limited resources in the free tier and the heavy resource consumption of Keras.
#This is a workaround to limit resource consumption

from keras import backend as K

K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))


# In[ ]:


import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import LeakyReLU

from keras import backend as K
import numpy as np


# In[ ]:


batch_size = 128
num_classes = 10
epochs = 1

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# # Training a simple model
# First we'll train a simple softmax regressor and check what accuracy we get

# In[ ]:


model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Dense(num_classes, activation='softmax'))



model.compile(loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy'])

model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test))
        
score = model.evaluate(x_test, y_test, verbose=0)

print('\n')
print('Accuracy:',score[1])


# In[ ]:


#some cleanup from the previous run
get_ipython().system('rm -f ker_*')
get_ipython().system('rm -f my_best_model.tgz')


# You should see an accuracy of approximately 90%. Now lets define a hyper-parameter grid including different activation functions and gradient descent optimizers. We’re optimizing over the grid using grid search (nested for loops) and store each model variant in a file. We then decide for the best one in order to deploy to IBM Watson Machine Learning.

# In[ ]:


#define parameter grid

activation_functions_layer_1 = ['sigmoid','tanh','relu']
opimizers = ['rmsprop','adagrad','adadelta']

#optimize over parameter grid (grid search)

for activation_function_layer_1 in activation_functions_layer_1:
    for opimizer in opimizers:
        
        model = Sequential()
        model.add(Dense(512, activation = activation_function_layer_1, input_shape=(784,)))
        model.add(Dense(num_classes, activation='softmax'))



        model.compile(loss='categorical_crossentropy',
              optimizer=opimizer,
              metrics=['accuracy'])

        model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
        
        score = model.evaluate(x_test, y_test, verbose=0)
        save_path = "ker_func_mnist_model_2.%s.%s.%s.h5" % (activation_function_layer_1,opimizer,score[1])
        model.save(save_path)


# # Model evaluation
# Let's have a look at all the models and see which hyper parameter configuration was the best one. You should see that relu and rmsprop gives you > 95% of accuracy on the validation set

# In[ ]:


ls -ltr ker_*


# Now it's time to create a tarball out of your favorite model, please replace the name of your favorite model H5 file with “please-put-me-here”

# In[ ]:


get_ipython().system('tar -zcvf my_best_model.tgz please-put-me-here.h5')


# ## 2.0 Save the trained model to WML Repository

# We will use `watson_machine_learning_client` python library to save the trained model to WML Repository, to deploy the saved model and to make predictions using the deployed model.</br>
# 
# 
# `watson_machine_learning_client` can be installed using the following `pip` command in case you are running outside Watson Studio:
# 
# `!pip install watson-machine-learning-client --upgrade`

# In[ ]:


from watson_machine_learning_client import WatsonMachineLearningAPIClient


# Please go to https://cloud.ibm.com/,  login, click on the “Create Resource” button. From the “AI” category, please choose “Machine Learning”. Wait for the “Create” button to activate and click on “Create”. Click on “Service Credentials”, then “New Credential”, then “Add”. From the new entry in the table, under “ACTIONS”, please click on “View Credentials”. Please copy the whole JSON object to your clipboard. Now just paste the JSON object below so that you are able to use your personal instance of Watson Machine Learning.

# In[ ]:


wml_credentials={
  "apikey": "hZ00Ov4tpXF5rzRUEyObEC7J1f_4Cvu8mkoYRh9AtHdL",
  "iam_apikey_description": "Auto generated apikey during resource-key operation for Instance - crn:v1:bluemix:public:pm-20:us-south:a/4b5f219cdaee498f9dac672a8966c254:708f4e4e-ffa6-4be2-8427-7a0a73ae6949::",
  "iam_apikey_name": "auto-generated-apikey-ae8c30a4-8f83-44e2-98b5-9461e847b11f",
  "iam_role_crn": "crn:v1:bluemix:public:iam::::serviceRole:Writer",
  "iam_serviceid_crn": "crn:v1:bluemix:public:iam-identity::a/4b5f219cdaee498f9dac672a8966c254::serviceid:ServiceId-c6a23b0b-5e7d-47b0-a3e0-6a2b51aa1817",
  "instance_id": "708f4e4e-ffa6-4be2-8427-7a0a73ae6949",
  "password": "",
  "url": "https://us-south.ml.cloud.ibm.com",
  "username": "ae8c30a4-8f83-44e2-98b5-9461e847b11f"
}


# In[ ]:


client = WatsonMachineLearningAPIClient(wml_credentials)


# In[ ]:


model_props = {client.repository.ModelMetaNames.AUTHOR_NAME: "IBM", 
               client.repository.ModelMetaNames.AUTHOR_EMAIL: "ibm@ibm.com", 
               client.repository.ModelMetaNames.NAME: "KK3_clt_keras_mnist",
               client.repository.ModelMetaNames.FRAMEWORK_NAME: "tensorflow",
               client.repository.ModelMetaNames.FRAMEWORK_VERSION: "1.5" ,
               client.repository.ModelMetaNames.FRAMEWORK_LIBRARIES: [{"name": "keras", "version": "2.1.3"}]
              }


# In[ ]:


published_model = client.repository.store_model(model="my_best_model.tgz", meta_props=model_props)


# In[ ]:


published_model_uid = client.repository.get_model_uid(published_model)
model_details = client.repository.get_details(published_model_uid)


# ## 3.0 Deploy the Keras model

# In[ ]:


client.deployments.list()


# To keep your environment clean, just delete all deployments from previous runs

# In[ ]:


client.deployments.delete("PASTE_YOUR_GUID_HERE_IF_APPLICABLE")


# In[ ]:


created_deployment = client.deployments.create(published_model_uid, name="k1_keras_mnist_clt1")


# ## Test the model

# In[ ]:


#scoring_endpoint = client.deployments.get_scoring_url(created_deployment)
scoring_endpoint = created_deployment['entity']['scoring_url']
print(scoring_endpoint)


# In[ ]:


x_score_1 = x_test[23].tolist()
print('The answer should be: ',np.argmax(y_test[23]))
scoring_payload = {'values': [x_score_1]}


# In[ ]:


predictions = client.deployments.score(scoring_endpoint, scoring_payload)
print('And the answer is!... ',predictions['values'][0][1])

