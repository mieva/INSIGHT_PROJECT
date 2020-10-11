## App to upload an image and run the NN to detect obejcts drawing bbxs 
### and returning the number of detected objects

import streamlit as st
import os

# Image processing
from PIL import Image
from cv2 import rectangle

import argparse

import torch
from FruitOnTreeDetection.model_definition import get_instance_frcnn_model
from FruitOnTreeDetection.predictions_rcnn import test_apple_detection


# Uploading an image
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# Display the selected image
def display_image(filename):
    # Use PIL
    img = Image.open(filename)
    st.image(filename, width=200, caption='Selected image')


def model_prediction(image, model):
    """
    Run the model to detect objects

    :param image:
    :param model:
    :return:
    """
    img, prediction = test_apple_detection(image, model)
    # show the image
    st.image(img, width=200, caption='Detected objects')
    num_obj = len(prediction['boxes'])
    return num_obj


def main(weights_path):
    # Title
    st.title('Fruit On Tree Detection App')

    # Ask to select a file
    filename = file_selector('./Images/')
    st.write('You selected `%s`' % filename)

    # Display selected image
    display_image(filename)

    # Button to run the model
    if st.button("Run detection"):
        # Now let's instantiate the model
        model = get_instance_frcnn_model(num_classes=2)
        # Upload weights (depend on device, cuda or cpu)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))

        # Display the annotated image by running the model.
        num_detected_fruit = model_prediction(filename, model)

        apple_mean_weight = 0.3
        st.write('Estimated yield is `%.2f` kg' % (num_detected_fruit*apple_mean_weight))


if __name__ == "__main__":

    weights_path = './models/model_12.pth'

    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dir', type=str, default=root_dir,
    #                     help='path to folder containing subfolders images and masks')
    parser.add_argument('--weights_path', type=str, default=weights_path,
                        help='path to folder containing model weights')
    args = parser.parse_args()

    main(args.weights_path)
