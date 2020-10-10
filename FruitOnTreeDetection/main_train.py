import argparse

from FruitOnTreeDetection.training_rcnn import train_rcnn, training_evaluation
from FruitOnTreeDetection.dataset_splitting import data_preparation

root_dir = '/content/drive/My Drive/INSIGHTPROGRAM/Data/detection/detection/train/'
output_dir = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/'
weights_path = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/model_frozen_19.pth'


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default=root_dir,
                        help='path to folder containing subfolders images and masks')
    parser.add_argument('--output_dir', type=str, default=output_dir,
                        help='path to folder to save training output')
    parser.add_argument('--weights_path', type=str, default=weights_path,
                        help='path to folder containing model weights')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--splitting_index', type=int, default=-200,
                        help='Dataset splitting index')
    args = parser.parse_args()

    data_loader, data_loader_test = data_preparation(args.root_dir, args.splitting_index)
    model = train_rcnn(data_loader, args.output_dir, args.num_epochs, args.weights_path)
    training_evaluation(model, data_loader, data_loader_test)

