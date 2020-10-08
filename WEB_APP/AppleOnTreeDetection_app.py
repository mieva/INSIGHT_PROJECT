## App to upload an image and run the NN to detect obejcts drawing bbxs 
### and returning the number of detected objects

import streamlit as st
import pandas as pd
import numpy as np
import os, urllib


# Image processing
from PIL import Image
import cv2

from cv2 import imread
from cv2 import imshow
from cv2 import rectangle

import torch
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Uploading an image
def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)


# Display the selected image
def display_image(filename):
	# Use PIL
	img = Image.open(filename)
	st.image(filename,width=200,caption='Selected image')

# Defying an istance for Resnet50 
def get_instance_frcnn_model(num_classes):
  # load a model pre-trained pre-trained on COCO
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

  # freeze parameters from the most inner layers...
  #for param in model.parameters():
  #    param.requires_grad = False
 
  # replace the classifier with a new one, that has
  # num_classes which is user-defined
  num_classes = 2  # 1 class (apple) + background
  # get number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

  return model

# Run the model to detect objects
def test_model_prediction(image, model):
	# Check the device
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	# put the model in evaluation mode
	if torch.cuda.is_available():
		model.cuda()
	model.eval()
	# Tranform the png
	#single_loaded_img = img
	#single_loaded_img = single_loaded_img.to(device)
	#single_loaded_img = single_loaded_img[None, None]
	#single_loaded_img = single_loaded_img.type('torch.FloatTensor')
	image = Image.open(image).convert("RGB")
	image = transforms.ToTensor()(image)
	# Run the model
	with torch.no_grad():
		#prediction = model(single_loaded_img)
		prediction = model([image.to(device)])[0]

	# print bounding box for each detected object
	img = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
	img = np.array(img)
	for box in prediction['boxes']:
		#extract
		#x, y, width, height = box
		#x2, y2 = x + width, y + height
		x, y, x2, y2 = box
		# draw a rectangle over the pixels
		rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (255,0,0), 2)
		# draw a rectangle over the pixels
		#rectangle(img, (x, y), (x2, y2), (255,0,0), 1)
		#sample = cv2.rectangle(img, pt1=(y, x), pt2=(y2, x2), color=(0, 0, 255), thickness=3)

	#ax.set_axis_off()
	#im = ax.imshow(sample)
	#st.pyplot()
	#st.write("# Results")

	# show the image
	st.image(img,width=200,caption='Detected objects')

	# Number of detect obj
	num_obj = len(prediction['boxes'])
	#st.write(num_obj)

	return num_obj

def main():

	# Title
	st.title('Fruit On Tree Detection App')

	# Ask to select a file
	filename = file_selector()
	st.write('You selected `%s`' % filename)

	# Display selected image
	display_image(filename)

	# Button to run the model
	#st.button("Run detection")
	if st.button("Run detection"):

		# Now let's instantiate the model
		# our dataset has two classes only - background and apple
		num_classes = 2
		# get the model using our helper function
		model = get_instance_frcnn_model(num_classes)
		#Upload weights (depend on device, cuda or cpu)
		#model.load_state_dict(torch.load('my_model_weights.pth', map_location=torch.device('cpu')))
		model.load_state_dict(torch.load('model_3.pth', map_location=torch.device('cpu')))
		
		
		# Display the annotated image by running the model.
		num_detected_fruit = test_model_prediction(filename, model)

		st.write('Detected `%d` objects' % num_detected_fruit)


# External files to download.
#EXTERNAL_DEPENDENCIES = {
#    "yolov3.weights": {
#        "url": "https://pjreddie.com/media/files/yolov3.weights",
#        "size": 248007048
#    },
#    "yolov3.cfg": {
#        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
#        "size": 8342
#    }
#}




if __name__ == "__main__":
    main()



