## App to upload an image and run the NN to detect obejcts drawing bbxs 
### and returning the number of detected objects

import streamlit as st
import os

import pandas as pd
import math

# Image processing
from PIL import Image

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
    return num_obj, prediction

def get_r_componets(df):
  # Function to calculate x_max - x_min, y_max - y_min
  x_max = df['p_x2'].max()
  x_min = df['p_x1'].min()
  d1 = x_max - x_min

  y_max = df['p_y2'].max()
  y_min = df['p_y1'].min()
  d2 = y_max - y_min

  d_mean = (d1 + d2)/2

  # Mean dimension of a bbx
  df['dx'] = df['p_x2'] - df['p_x1']
  bbx_size_mean = df['dx'].mean()
  bbx_size_std = df['dx'].std()

  return d_mean, bbx_size_mean, bbx_size_std

def get_form_factor(dmean, bbx_size_mean, bbx_size_std):
  # Function to create a df with the form factor estimation first verion
  # df = pd.DataFrame(data=[dmean, bbx_size_mean, bbx_size_std]).T
  # df.columns = ['dmean', 'bbx_size_mean', 'bbx_size_std']
  #df['form_factor'] = (df['d1']*df['d1'] + df['d2']*df['d2']).apply(lambda x: 2/3*math.sqrt(x))
  #df['form_factor'] = (2/3*(dmean/2))/(bbx_size_mean+bbx_size_std)
  form_factor = (1/3*(dmean/2))/(bbx_size_mean+bbx_size_std)

  return form_factor


def main(weights_path):

    avg_prec = 0.71

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
        num_detected_fruit, prediction = model_prediction(filename, model)

        # creating a Dataframe object
        df_prediction = pd.DataFrame(prediction)

        # take the r components
        d_mean, bbx_size_mean, bbx_size_std = get_r_componets(df_prediction)

        # form factor
        form_factor = get_form_factor(d_mean, bbx_size_mean, bbx_size_std)

        # Correct the detected number of fruit by avg precision
        num_detected_fruit_corrected = num_detected_fruit*avg_prec

        # Estimated number of fruits on the whoole tree
        estimated_fruit_on_tree = num_detected_fruit_corrected*form_factor

        st.write('Number of detected fruit in the image is `%d`' % num_detected_fruit_corrected)
        st.write('Estimated number of fruits on the whole tree is `%d`' % estimated_fruit_on_tree)
        #st.write('Estimated yield is `%.2f` kg' % (num_detected_fruit*apple_mean_weight))


if __name__ == "__main__":

    weights_path = './models/model_12.pth'

    parser = argparse.ArgumentParser()
    # parser.add_argument('--root_dir', type=str, default=root_dir,
    #                     help='path to folder containing subfolders images and masks')
    parser.add_argument('--weights_path', type=str, default=weights_path,
                        help='path to folder containing model weights')
    args = parser.parse_args()

    main(args.weights_path)
