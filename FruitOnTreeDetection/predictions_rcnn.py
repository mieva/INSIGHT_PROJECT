from PIL import Image
import cv2
import numpy as np

from cv2 import rectangle
from matplotlib import pyplot as plt
import torch
from torchvision import transforms

output_file = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/model_predictions.csv'
weights_file = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/model_12.pth'


def make_predictions_on_dataset(dataset, model, output_file):
    """
    Function to run the model on the dataset_test and create an output file with predictions
      param dataset: dataset_test
      param model: the pytorch NN model
      param file_out: output file
    """
    model.eval()

    # Set the device
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cpu')

    data_loader_test = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                   shuffle=False, num_workers=1,
                                                   collate_fn=utils.collate_fn)

    # Predict on bboxes on each image
    f = open(output_file, 'a')
    for image, targets in data_loader_test:
        image = list(img.to(device) for img in image)
        outputs = model(image)
        for ii, output in enumerate(outputs):
            img_id = targets[ii]['image_id']
            img_name = dataset.dataset.get_img_name(img_id)
            #   print("Predicting on image: {}".format(img_name))
            boxes = output['boxes'].detach().numpy()
            scores = output['scores'].detach().numpy()

            im_names = np.repeat(img_name, len(boxes), axis=0)
            stacked = np.hstack((im_names.reshape(len(scores), 1), boxes.astype(int), scores.reshape(len(scores), 1)))

            # File to write predictions to
            np.savetxt(f, stacked, fmt='%s', delimiter=',', newline='\n')


def make_gt_file(dataset, output_file):
    """
    Function to write the GT into a .csv with a similar structure than predictions
    param dataset: dataset (all), GT
    param file_out: output file
    """
    f = open(output_file, 'a')
    for image, target in dataset:
        img_id = target['image_id']
        img_name = dataset.dataset.get_img_name(img_id)
        # print("Saving on image: {}".format(img_name))
        boxes = target['boxes'].detach().numpy()

        im_names = np.repeat(img_name, len(boxes), axis=0)
        stacked = np.hstack((im_names.reshape(len(boxes), 1), boxes.astype(int)))
        np.savetxt(f, stacked, fmt='%s', delimiter=',', newline='\n')


def test_apple_detection(image, model):
    """
    Draw bounding boxes on a single image

    param image: path of a single image
    param model: the pytorch NN model
    """
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    image = Image.open(image).convert("RGB")
    image = transforms.ToTensor()(image)
    print('Image size', image.size())

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with torch.no_grad():
        prediction = model([image.to(device)])[0]

    # print bounding box for each detected object
    img = Image.fromarray(image.mul(255).permute(1, 2, 0).byte().numpy())
    img = np.array(img)
    print('Image size', img.shape)
    for box in prediction['boxes']:
        # extract
        x, y, x2, y2 = box
        # draw a rectangle over the pixels
        rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (255, 0, 0), 2)

    return img, prediction

    # show the image
    # fig = plt.figure(figsize=(9, 12))
    # plt.axis("off")
    # plt.imshow(img)

    # Save example
    # plt.savefig('/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/single_fig_preds.png')
    # plt.close(fig)
