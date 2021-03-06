import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
from .core.grad_cam import GradCAM
from keras.applications import xception
from tensorflow.python.ops.numpy_ops import np_config
import PIL

TARGET_SIZE = 224
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

def format_image(image):
    return tf.image.resize(image, [TARGET_SIZE, TARGET_SIZE]) / 255.0


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


def plot_GradCAM(class_name, img):
    """Generate GradCAM of Uploaded image

    Args:
        class_name: The name of the predicted class
        img [jpg]: The file to generate the GradCAM for

    """

    # The name of the last layer provided that we will use for GradCAM output from Xception model
    last_conv_layer = 'block14_sepconv2_act'

    # Converting the class name to a numeric value
    class_index = [i for i, x in enumerate(class_names) if x == class_name]
    class_index = class_index[0]

    # allowing the tensor to run eagerly for tf.numpy()
    np_config.enable_numpy_behavior()

    # Creating the model used for prediction
    model = CreateModel()

    # Making sure to start from the beginning of the image and reading it
    img.seek(0)
    fimage = img.read()
    npimg = np.fromstring(fimage, np.uint8)
    input_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preparing the image for our model based on Xception model and setting the expected size
    img_array = xception.preprocess_input(input_img)
    img_array = format_image(img_array)

    # Defining the GradCAM object
    explainer = GradCAM()
    heat_map = explainer.explain(([img_array], None), model.layers[1], layer_name=last_conv_layer, class_index=class_index,  image_weight=.9 )

    # Saving the output to disk
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/images/GradCAMRaw.png')
    plt.imsave(path, heat_map, cmap='viridis')

    # Importing the Image library
    try:
        from PIL import Image
    except ImportError:
        import Image

    # Setting up the "original" image and the GradCAM output
    background = Image.open(ROOT_DIR + '/static/images/output.png')
    overlay = Image.open(ROOT_DIR + '/static/images/GradCAMRaw.png')

    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")

    # Overlaying the GradCAM output onto the original level and setting the opacity
    background = background.resize((TARGET_SIZE,TARGET_SIZE), resample=PIL.Image.LANCZOS)
    overlay = overlay.resize((TARGET_SIZE,TARGET_SIZE), resample=PIL.Image.LANCZOS )

    new_img = Image.blend(background, overlay, 0.4)
    new_img.save(ROOT_DIR + '/static/images/GradCAM.png')


def CreateModel():
    """Creation of the Convultional Neural Network used to predict the OCT images"""

    # Setting up the weights to be used from training previously
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(ROOT_DIR + '/static/model/')
    saved_weight_file = 'oct_singan.h5'

    # Creating the model based on Xception Network
    input_layer = tf.keras.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
    base_model =  xception.Xception(include_top=False, weights="imagenet", input_shape=(TARGET_SIZE, TARGET_SIZE, 3))
    base_model.trainable = True

    x = base_model(input_layer)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Dense(4, activation="softmax")(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy',
                  metrics=["accuracy"])

    # Now that the model is created we load the previously saved weights from our training run
    model.load_weights(path + saved_weight_file)

    return model