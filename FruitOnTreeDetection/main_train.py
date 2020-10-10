import argparse

from FruitOnTreeDetection.dataset_splitting import data_preparation, dataloader
from FruitOnTreeDetection.training_rcnn import train_rcnn, training_evaluation
from FruitOnTreeDetection.predictions_rcnn import make_predictions_on_dataset, make_gt_file
from FruitOnTreeDetection.evaluation_rcnn import make_evaluation_on_dataset


root_dir = '/content/drive/My Drive/INSIGHTPROGRAM/Data/detection/detection/train/'
output_dir = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/'
weights_path = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/model_frozen_19.pth'
ground_truth_file = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/model_gt.csv'
prediction_file = '/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/model_predictions.csv'


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
    parser.add_argument('--prediction_file', type=str, default=prediction_file,
                        help='output_file for predictions')
    parser.add_argument('--ground_truth_file', type=str, default=ground_truth_file,
                        help='output_file from the ground_truth')
    args = parser.parse_args()

    # Dataset
    dataset, dataset_test = data_preparation(args.root_dir, args.splitting_index)
    data_loader, data_loader_test = dataloader(dataset, dataset_test)
    # Start train
    model = train_rcnn(data_loader, args.output_dir, args.num_epochs, args.weights_path)
    training_evaluation(model, data_loader, data_loader_test)

    # Start predictions
    make_predictions_on_dataset(dataset_test, model, args.prediction_file)

    # File from ground truth
    make_gt_file(dataset, args.ground_truth_file)

    # Start evaluation
    make_evaluation_on_dataset(args.ground_truth_file, args.prediction_file)

