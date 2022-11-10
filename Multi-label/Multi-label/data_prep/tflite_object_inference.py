import tensorflow as tf 
import numpy as np
from PIL import Image
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob

interpreter = tf.lite.Interpreter(model_path='./tflite_object/model.tflite')
# Get input and output tensors.

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

float_input = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Randomly select test images
# images_to_test = random.sample(images, num_test_images)

# Loop over every image and perform detection
# for image_path in images_to_test:

# Load image and resize to expected shape [1xHxWx3]
# img_path = './1.png'
# /home/ai-team/Object_detection_models/data/test_data/animals/buffallo
for image_path in glob.glob('/home/ai-team/Object_detection_models/data/test_data/animals/dog/*'):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (640,640))
    input_data = np.expand_dims(image_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    if float_input:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Perform the actual detection by running the model with the image as input
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[1]['index'])[0] # Bounding box coordinates of detected objects
    classes = interpreter.get_tensor(output_details[3]['index'])[0] # Class index of detected objects
    scores = interpreter.get_tensor(output_details[0]['index'])[0] # Confidence of detected objects
    print(boxes)
    print("CLasses:",classes)
    print("scores:",scores)
    class_names  = ["hen","dog","goat","cat","cow","buffalo","sheep","pig"]
    predict = np.argmax(scores)
    print(class_names[predict])

