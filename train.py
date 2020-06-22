import tensorflow as tf
import os
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import models,layers
from tensorflow.keras.optimizers import RMSprop,SGD, Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau





root = "E:/Emotion Recognition/dataset/"




def data_preprocess(df):
    k=int(0.7*len(df))
    df1=df[:k]
    df2=df[k:]
    y_t=df1['emotion']
    y_t=np.array(y_t)
    y_v=df2['emotion']
    y_v=np.array(y_v)
    x_t=df1.pixels.apply(lambda x: pd.Series(str(x).split(" ")))
    x_t=np.array(x_t)
    x_t=x_t.astype(int)
    x_v=df2.pixels.apply(lambda x: pd.Series(str(x).split(" ")))
    x_v=np.array(x_v)
    x_v=x_v.astype(int)
    return x_t,y_t,x_v,y_v
    

    
data = pd.read_csv(root+"fer_2013.csv")


X_train, Y_train, X_test, Y_test = data_preprocess(data)

X_train = (1.*X_train)/255

X_test = (1.*X_test)/255


############################################################


num_classes= 7
r, c = 48,48

batch_size = 32

X_train=X_train.reshape((20096,48,48,1))
X_test=X_test.reshape((8613,48,48,1))


model = models.Sequential()

#Input layer

model.add(layers.Conv2D(32,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal', input_shape = (r,c,1)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal', input_shape = (r,c,1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))


#Hidden Layer 1

model.add(layers.Conv2D(64,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

#Hidden Layer 2

model.add(layers.Conv2D(128,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

#Hidden Layer 3

model.add(layers.Conv2D(256,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256,(3,3), padding = 'same', activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Dropout(0.2))

#Hidden Layer 4

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))


#Hidden Layer 2

model.add(layers.Dense(64, activation = 'elu', kernel_initializer = 'he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

#Final Layer
model.add(layers.Dense(num_classes, activation = 'softmax', kernel_initializer='he_normal'))

model.summary()

opt=RMSprop(learning_rate= 0.001)
model.compile(optimizer = opt, loss= "sparse_categorical_crossentropy", metrics = ['accuracy'])

history = model.fit(X_train, Y_train, epochs = 100, batch_size=batch_size, validation_data=(X_test,Y_test))



# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")




json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = models.model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


