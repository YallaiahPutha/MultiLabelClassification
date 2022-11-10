
from gc import callbacks
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import tensorflow
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

df=pd.read_csv('./annotate.csv')

img_width=224
img_height=224

X=[]
cnt=0
# for i in tqdm(range(df.shape[0])):
#     path='/home/ai-team/Object_detection_models/Vegetable_model/'+df['ImageId'][i]
#     img=cv2.imread(path)
#     img=cv2.resize(img,(224,224))
#     img=np.asarray(img)
#     img = img/255.0
#     X.append(img)

for i in tqdm(range(df.shape[0])):
    path='/home/ai-team/Object_detection_models/Vegetable_model/'+df['ImageId'][i]
    img = load_img(path,target_size=(img_width,img_height,3))
    img = img_to_array(img)
    img = img/255.0
    X.append(img)

X=np.array(X)
print(X.shape)

#**********************************************************************************

# y=df.columns.values[2:]
# print(y)
y=df.drop(['ImageId','Labels','Unnamed: 0'],axis=1)
print(y)
y=y.to_numpy()
print(y.shape)
# #***********************************************************************************

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0, test_size = 0.15)
print("Shape of x_traain",X_train.shape)

model=Sequential()
model.add(MobileNetV2(include_top = False, weights="imagenet", input_shape=(224, 224, 3)))
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(13, activation = 'sigmoid'))
model.layers[0].trainable = False
model.summary()

model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
# early_stop=EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=5)
# ,callbacks=[early_stop]
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))
model.save('/home/ai-team/Object_detection_models/multilabel/save_model/')