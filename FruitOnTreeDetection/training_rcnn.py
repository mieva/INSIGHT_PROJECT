import os
import time
import datetime

from matplotlib import pyplot as plt
import pickle
import numpy as np
import torch

from FruitOnTreeDetection.model_definition import get_instance_frcnn_model
from Vision.engine import train_one_epoch, evaluate


def save_metrics(metrics, file_path):
    """
    Save the metrics produced during the train in a pickle file

    :param metrics: list of metric_loggers
    :param file_path: file that stores the data
    """
    with open(file_path, 'wb') as f:
        pickle.dump(metrics, f)

def open_metrics(file_path):
    with open(file_path, 'rb') as f:
        metrics = pickle.load(f)
    return metrics


def plot_one_metric(metrics, metric_name, metric_value='median'):
    """
    Plot one of the metrics produced during the training

    :param metrics: list of metric_loggers
    :param metric_name: the name of the metric to plot
    :param metric_value: the property of the metric to plot. Either total,
                         count, avg, median, global_avg, max, or value.
    """
    metric = np.array([m.__getattr__(metric_name).__getattribute__(metric_value) for m in metrics])
    plt.plot(np.arange(metric.size), metric)
    plt.xlabel('epoch')
    plt.ylabel('%s %s' % (metric_name, metric_value))
    plt.title('%s %s' % (metric_name, metric_value))
    # plt.show()


def plot_train_metrics(epoch_metrics):
    fig = plt.figure(figsize=[12, 9], tight_layout=True)
    plt.subplot(2, 1, 1)
    plot_one_metric(epoch_metrics, 'loss')
    plt.subplot(2, 4, 5)
    plot_one_metric(epoch_metrics, 'loss_box_reg')
    plt.subplot(2, 4, 6)
    plot_one_metric(epoch_metrics, 'loss_classifier')
    plt.subplot(2, 4, 7)
    plot_one_metric(epoch_metrics, 'loss_objectness')
    plt.subplot(2, 4, 8)
    plot_one_metric(epoch_metrics, 'loss_rpn_box_reg')

    plt.show()


def train_rcnn(data_loader, output_dir, num_epochs=10, weights_path=None):
    # Now let's instantiate the model and the optimizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # SET THE MODEL TO TRAIN
    model = get_instance_frcnn_model(num_classes=2, freeze=False)
    if weights_path:
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(weights_path))
        else:
            model.load_state_dict(torch.load(weights_path), map_location=torch.device('cpu'))

    #model = get_instance_frcnn_model(num_classes, freeze=True)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=0.005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    print("Start training")
    start_time = time.time()

    epoch_metrics = []
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        epoch_metrics.append(metric_logger)
        # update the learning rate
        lr_scheduler.step()

        if epoch % 4 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, 'model_frozen_{}.pth'.format(epoch)))

    # Save the last learning output on a file
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_frozen_{}.pth'.format(num_epochs-1)))

    # print('\nEvaluate on training dataset')
    # evaluate(model, data_loader, device=device)

    # print('\nEvaluate on test dataset')
    # evaluate(model, data_loader_test, device=device)

    # Check how long it takes
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    save_metrics(epoch_metrics, output_dir+'train_metrics.pkl')
    plot_train_metrics(epoch_metrics)

    return model


def training_evaluation(model, data_loader, data_loader_test):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('\nEvaluate on training dataset')
    evaluate(model, data_loader, device=device)

    print('\nEvaluate on test dataset')
    evaluate(model, data_loader_test, device=device)
