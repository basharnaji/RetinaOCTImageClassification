import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from tf_explain.core.grad_cam import GradCAM
from keras.applications import xception

def format_image(image):
    return tf.image.resize(image, [224, 224]) / 255.0

def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/model/')
    tflite_model_file = 'XceptionSinGANOCT.tflite'

    # Load TFLite model and allocate tensors.
    with open(path + tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Gets model input and output details.
    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    fimage = img.read()
    npimg = np.fromstring(fimage, np.uint8)
    input_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    format_img = format_image(input_img)

    format_img = np.expand_dims(format_img, axis=0)

    # Sets the value of the input tensor
    interpreter.set_tensor(input_index["index"], format_img)
    # Invoke the interpreter.
    interpreter.invoke()

    predictions_array = interpreter.get_tensor(output_index["index"])
    predicted_label = np.argmax(predictions_array)

    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    return class_names[predicted_label]


def plot_category(img):
    """Plot the input image

    Args:
        img [jpg]: image file
    """

    read_img = mpimg.imread(img)
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/images/output.png')
    plt.imsave(path, read_img, cmap='gray')

def plot_heatmap(class_name, img):

    last_conv_layer = 'block14_sepconv2_act'
    class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    class_index = [i for i, x in enumerate(class_names) if x == class_name]

    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/model/')
    tflite_model_file = 'XceptionBase.tflite'

    # Load TFLite model and allocate tensors.
    with open(path + tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Gets model input and output details.
    input_index = interpreter.get_input_details()[0]
    output_index = interpreter.get_output_details()[0]
    img.seek(0)
    fimage = img.read()
    npimg = np.fromstring(fimage, np.uint8)
    input_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    format_img = xception.preprocess_input(input_img)
    format_img = format_image(format_img)
    format_img = np.expand_dims(format_img, axis=0)

    interpreter.set_tensor(input_index['index'], format_img)
    interpreter.invoke()

    print ('=============\n', format_img.shape , '\n================')
    print ('************\n', type(interpreter.get_tensor(output_index["index"])), '\n*************' )

    explainer = GradCAM()
    heat_map = explainer.explain(format_img, interpreter.get_tensor(output_index["index"]), layer_name=last_conv_layer, class_index=class_index, image_weight=0.9)
    plt.imshow(heat_map)
    plt.show()