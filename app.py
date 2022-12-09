import streamlit as st

import cv2
import matplotlib.pyplot as plt 
import os
import numpy as np
import datetime
import itertools
import h5py
import io
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.models import Model

st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def load_cnn1():
    model_ = load_model('my_model.h5')
    return model_

def preprocessed_image(file):
    image = file.resize((50,50), Image.ANTIALIAS)
    image = np.array(image)
    image = np.expand_dims(image, axis=0) 
    return image

def main():
    st.title('CNN for Classification Malaria Cells')
    st.sidebar.title('Web Apps using Streamlit')
    st.sidebar.text(""" Project to detect the malaria-infected image using CNN Algorithm - by Aakash Haridas""")
    menu = {1:"Home",2:"Perform Prediction"}
    def format_func(option):
        return menu[option]
    choice= st.sidebar.selectbox("Menu",options=list(menu.keys()), format_func=format_func)
    if choice == 1 :
        st.subheader("Dataset Malaria Cells")
        st.markdown("#### Preliminary")
        """ 
        
        This is datasets of segmented cells from the thin blood smear slide images from the Malaria Screener research activity.
        The Dataset is obtained from researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC),
        part of National Library of Medicine (NLM), that developed a mobile application that runs on a standard Android smartphone attached to a conventional light microscope. 
        Giemsa-stained thin blood smear slides from 150 P. falciparum-infected and 50 healthy patients were collected and photographed 
        at Chittagong Medical College Hospital, Bangladesh. The smartphone’s built-in camera acquired images of slides for each microscopic field of view. 
        The images were manually annotated by an expert slide reader at the Mahidol-Oxford Tropical Medicine Research Unit in Bangkok, Thailand. 
        The de-identified images and annotations are archived at NLM (IRB#12972). then applied a level-set based algorithm to detect and segment the red blood cells. 
        The dataset contains a total of 27,558 cell images with equal instances of parasitized and uninfected cells 
        """
        
        st.markdown("#### Previous research")
        """ 
        
        The data appear along with the publications : 
        
        Rajaraman S, Antani SK, Poostchi M, Silamut K, Hossain MA, Maude, RJ, Jaeger S, Thoma GR. (2018) 
        Pre-trained convolutional neural networks as feature extractors toward improved Malaria parasite detection in thin blood smear images.
        
        link : https://peerj.com/articles/4568/ 
        
        Rajaraman S, Jaeger S, Antani SK. (2019) Performance evaluation of deep neural ensembles toward malaria parasite detection in thin-blood smear images
        
        link : https://peerj.com/articles/6977/
        """
        
        st.markdown("#### Links for Malaria Dataset")
        """ 
        More information and download links for this Dataset provided below
        
        https://lhncbc.nlm.nih.gov/publication/pub9932
        
        https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria
        """

    elif choice == 2 :
        st.subheader("CNN Model")
        st.markdown("#### Simple CNN")
        model_1 = load_cnn1()
        st.subheader('Test on an Image')
        images = st.file_uploader('Upload Image',type=['jpg','png','jpeg'])
        if images is not None:
                images = Image.open(images)
                st.text("Image Uploaded!")
                st.image(images,width=300)
                used_images = preprocessed_image(images)
                predictions = np.argmax(model_1.predict(used_images), axis=-1)
                if predictions == 1:
                    st.error("Cell is normal...The person is healthy")
                elif predictions == 0:
                    st.success("Cell is malaria parasitized...The person has Malaria")
                 
if __name__ == "__main__":
    main()
