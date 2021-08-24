import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os


def read_image(image):
    return mpimg.imread(image)


def format_image(image):
    return tf.image.resize(image[tf.newaxis, ...], [224, 224]) / 255.0


def get_category(img):
    """Write a Function to Predict the Class Name

    Args:
        img [jpg]: image file

    Returns:
        [str]: Prediction
    """
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/model/')
    tflite_model_file = 'converted_model.tflite'

    # Load TFLite model and allocate tensors.
    with open(path + tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Interpreter interface for TensorFlow Lite Models.
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Gets model input and output details.
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    input_img = read_image(img)
    format_img = format_image(input_img)
    # Sets the value of the input tensor
    interpreter.set_tensor(input_index, format_img)
    # Invoke the interpreter.
    interpreter.invoke()

    predictions_array = interpreter.get_tensor(output_index)
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
    plt.imsave(path, read_img)
