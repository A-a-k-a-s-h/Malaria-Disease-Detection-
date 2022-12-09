# -*- coding: utf-8 -*-
"""Malaria_Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1c1drqNrRUu3UJ2sr48e_11PrmDM7Dk8n

**Accessing Dataset from Kaggle**

**Importing Necessary libraries**
"""

import pandas as pd
#math operations
import numpy as np
#machine learning
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import cv2
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
            
from random import shuffle
from tqdm import tqdm  
import scipy
import skimage
from skimage.transform import resize
import random

# Colab's file access feature
from google.colab import files

#retrieve uploaded file
uploaded = files.upload()

#print results
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  
# Then move kaggle.json into the folder where the API expects to find it.
!mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets list

!kaggle datasets download -d iarunava/cell-images-for-detecting-malaria/downloads/cell-images-for-detecting-malaria.zip/

!ls
!unzip cell-images-for-detecting-malaria.zip
!ls

"""**Making a path for Parasitized and Uninfected Image Dataset** 


"""

PARA_DIR = "cell_images/Parasitized/"
UNIF_DIR =  "cell_images/Uninfected/"

Pimages = os.listdir(PARA_DIR)
Nimages = os.listdir(UNIF_DIR)

sample_normal = random.sample(Nimages,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('cell_images/Uninfected/'+sample_normal[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Uninfected')
plt.show()

sample_parasite = random.sample(Pimages,6)
f,ax = plt.subplots(2,3,figsize=(15,9))

for i in range(0,6):
    im = cv2.imread('cell_images/Parasitized/'+sample_parasite[i])
    ax[i//3,i%3].imshow(im)
    ax[i//3,i%3].axis('off')
f.suptitle('Parasitized')
plt.show()

data=[]
labels=[]
Parasitized=os.listdir("cell_images/Parasitized/")
for a in Parasitized:
    try:
        image=cv2.imread("cell_images/Parasitized/"+a)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")

Uninfected=os.listdir("cell_images/Uninfected/")
for b in Uninfected:
    try:
        image=cv2.imread("cell_images/Uninfected/"+b)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((50, 50))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")

Cells=np.array(data)
labels=np.array(labels)

np.save("Cells",Cells)
np.save("labels",labels)

Cells=np.load("Cells.npy")
labels=np.load("labels.npy")

s=np.arange(Cells.shape[0])
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

num_classes=len(np.unique(labels))
len_data=len(Cells)

"""**Normalizing the dataset**"""

(x_train,x_test)=Cells[(int)(0.1*len_data):],Cells[:(int)(0.1*len_data)]
x_train = x_train.astype('float32')/255 # As we are working on image data we are normalizing data by dividing 255.
x_test = x_test.astype('float32')/255
train_len=len(x_train)
test_len=len(x_test)

(y_train,y_test)=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]

#Doing One hot encoding as classifier has multiple classes
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

from keras.callbacks import EarlyStopping, ModelCheckpoint

# Set random seed
np.random.seed(0)

"""**CNN Model**"""

#creating sequential model
model=Sequential()
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu",input_shape=(50,50,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(512,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(2,activation="softmax"))#2 represent output layer neurons 
model.summary()

# compile the model with loss as categorical_crossentropy and using adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=2),
             ModelCheckpoint('.mdl_wts.hdf5', monitor='val_loss', save_best_only=True)]

"""**Training the Model**"""

#Fit the model with min batch size as 32 can tune batch size to some factor of 2^power ] 
h=model.fit(x_train,y_train,batch_size=32,callbacks=callbacks, validation_data=(x_test,y_test),epochs=20,verbose=1)

from numpy import loadtxt
from keras.models import load_model
model = load_model('.mdl_wts.hdf5')

score=model.evaluate(x_test,y_test)
print(score)

"""**Test Accuracy**"""

accuracy = model.evaluate(x_test, y_test, verbose=1)
print('\n', 'Test_Accuracy:-', accuracy[1])

from sklearn.metrics import confusion_matrix
pred = model.predict(x_test)
pred = np.argmax(pred,axis = 1) 
y_true = np.argmax(y_test,axis = 1)

"""**Evaluation Metrics**

**Confusion Matrix**
"""

CM = confusion_matrix(y_true, pred)
from mlxtend.plotting import plot_confusion_matrix
fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))
plt.show()

i=9
pred = model.predict(x_test,batch_size=1)
pred = np.argmax(pred,axis = 1)

pred[0]

"""**Saving the Model**"""

model.save('my_model.h5')

"""**Layers in the CNN Model**"""

len(model.layers)

x_test.shape[0]

"""**Model Accuracy**"""

plt.plot(h.history['accuracy'])
plt.plot(h.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Accuracy")
plt.ylabel("Epochs")
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

"""**ROC Curve**"""

from sklearn.metrics import auc
fpr_keras, tpr_keras, thresholds = roc_curve(y_true.ravel(), pred.ravel())
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

plot_roc_curve(fpr_keras, tpr_keras)

"""**Classification Report**"""

from sklearn.metrics import classification_report
print('{}'.format( 
                           classification_report(y_true , pred)))

# get predictions on the test set
y_hat = model.predict(x_test)

# define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
malaria_labels = ['Parasitized','Uninfected']

x_test.shape[0]

"""**Predicting random test images**"""

# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(20, 8))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=12, replace=False)):
    ax = fig.add_subplot(4,4, i+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_hat[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(malaria_labels[pred_idx], malaria_labels[true_idx]),
                 color=("blue" if pred_idx == true_idx else "orange"))

"""**Deploying the model as Web Application using Streamlit**"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st
# 
# import cv2
# import matplotlib.pyplot as plt 
# import os
# import numpy as np
# import datetime
# import itertools
# import h5py
# import io
# from PIL import Image
# import tensorflow as tf
# from keras.models import load_model
# from keras.models import Model
# 
# st.set_option('deprecation.showfileUploaderEncoding', False)
# 
# @st.cache(allow_output_mutation=True,suppress_st_warning=True)
# def load_cnn1():
#     model_ = load_model('/content/my_model.h5')
#     return model_
# 
# def preprocessed_image(file):
#     image = file.resize((50,50), Image.ANTIALIAS)
#     image = np.array(image)
#     image = np.expand_dims(image, axis=0) 
#     return image
# 
# def main():
#     st.title('CNN for Classification Malaria Cells')
#     st.sidebar.title('Web Apps using Streamlit')
#     st.sidebar.text(""" Project to visualize the CNN layers on malaria-infected image by Aakash Haridas""")
#     menu = {1:"Home",2:"Perform Prediction"}
#     def format_func(option):
#         return menu[option]
#     choice= st.sidebar.selectbox("Menu",options=list(menu.keys()), format_func=format_func)
#     if choice == 1 :
#         st.subheader("Dataset Malaria Cells")
#         st.markdown("#### Preliminary")
#         """ 
#         
#         This is datasets of segmented cells from the thin blood smear slide images from the Malaria Screener research activity.
#         The Dataset is obtained from researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC),
#         part of National Library of Medicine (NLM), that developed a mobile application that runs on a standard Android smartphone attached to a conventional light microscope. 
#         Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients were collected and photographed 
#         at Chittagong Medical College Hospital, Bangladesh. The smartphone’s built-in camera acquired images of slides for each microscopic field of view. 
#         The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand. 
#         The de-identified images and annotations are archived at NLM (IRB#12972). then applied a level-set based algorithm to detect and segment the red blood cells. 
#         The dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells 
#         """
#         
#         st.markdown("#### Previous research")
#         """ 
#         
#         The data appear along with the publications : 
#         
#         Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ, Jaeger S, Thoma GR. (2018) 
#         Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images.
#         
#         link : https://peerj.com/articles/4568/ 
#         
#         Rajaraman S, Jaeger S, Antani SK. (2019) Performance evaluation of deep neural ensembles toward malaria parasite detection in thin-blood smear images
#         
#         link : https://peerj.com/articles/6977/
#         """
#         
#         st.markdown("#### Links for Malaria Dataset")
#         """ 
#         More information and download links for this Dataset provided below
#         
#         https://lhncbc.nlm.nih.gov/publication/pub9932
#         
#         https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
#         """
# 
#     elif choice == 2 :
#         st.subheader("CNN Model")
#         st.markdown("#### Simple CNN")
#         model_1 = load_cnn1()
#         st.subheader('Test on an Image')
#         images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
#         if images is not None:
#                 images = Image.open(images)
#                 st.text("Image Uploaded!")
#                 st.image(images,width=300)
#                 used_images = preprocessed_image(images)
#                 predictions = np.argmax(model_1.predict(used_images), axis=-1)
#                 if predictions == 1:
#                     st.error("Cell is normal...The person is healthy")
#                 elif predictions == 0:
#                     st.success("Cell is malaria parasitized...The person has Malaria")
#                  
# if __name__ == "__main__":
#     main()

!ls

!pip install -q streamlit

!npm install localtunnel

!streamlit run /content/app.py &>/content/logs.txt &

"""**Creating a local tunnel for Web Application**"""

!npx localtunnel --port 8501