from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

img_width =224
img_height = 224

img = load_img('1.jpg', target_size=(img_width, img_height, 3))
plt.imshow(img)
img = img_to_array(img)
img = img/255.0
model = tf.keras.models.load_model("/home/ai-team/Object_detection_models/multilabel/save_model/")
img = img.reshape(1, img_width, img_height, 3)

classes = ["carrot","ginger","greenchilli","tomato","pumpkin","ladiesfinger","onion","bittergourd","brinjal","cucumber","garlic","potato","capsicum"]
print(classes)
y_prob = model.predict(img)
print(y_prob)
# result = np.argmax(y_prob[0])
# print(result)
top3 = np.argsort(y_prob[0])[:-4:-1]

for i in range(3):
  print(classes[top3[i]])