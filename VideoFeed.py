# -*- coding: utf-8 -*-
"""
Created on Wed May 13 19:25:19 2020

@author: shyam
"""



import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import models,layers
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import RMSprop,SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

class_label=['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cap = cv2.VideoCapture(0)

while True:
    ret, img=cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y) , (x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48), interpolation = cv2.INTER_AREA)
        
        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis = 0)

            
            preds = loaded_model.predict(roi)[0]
            label = class_label[preds.argmax()]
            cv2.rectangle(img, (x,y+h) , (x+w,y+h+30),(255,255,255),-1)
            cv2.putText(img,label,(x,y+h+18),cv2.FONT_ITALIC,w/250,(0,0,0),2)

        
        else:
            cv2.putText(img,"No faces found",(x,y+h+18),cv2.FONT_ITALIC,w/250,(0,0,0),2)
    
        
        
    cv2.imshow('img',img)
    k=cv2.waitKey(30) & 0xff
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
        