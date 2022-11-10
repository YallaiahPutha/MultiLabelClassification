import numpy as np
import tensorflow.lite as tflite
import cv2
import pathlib

classes = ["carrot","ginger","greenchilli","tomato","pumpkin","ladiesfinger","onion","bittergourd","brinjal","cucumber","garlic","potato","capsicum"]

# Load TFLite model and allocate tensors.
interpreter =tflite.Interpreter(model_path="./multilabel.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# input details
print(input_details)
# output details
print(output_details)

img_path = '1.jpg'
# read and resize the image
img = cv2.imread(img_path)
new_img1 = cv2.resize(img, (224, 224))
# images.append(new_img)
new_img2 = new_img1/255
new_img = np.float32(new_img2)
# resize the input tensor
input_tensor = np.array(np.expand_dims(new_img,0))

input_index =  interpreter.get_input_details()[0]['index']

interpreter.set_tensor(input_index,input_tensor)
interpreter.invoke()
output_details = interpreter.get_output_details()

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)

predict_label = np.argsort(output_data[0])[:-4:-1]
print(predict_label)

for i in range(3):
  print(classes[predict_label[i]])