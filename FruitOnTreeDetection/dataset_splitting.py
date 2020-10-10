import torch
from FruitOnTreeDetection.apple_dataset import AppleDataset
from FruitOnTreeDetection.Utilities.transforms import get_transform


root_dir = '/content/drive/My Drive/INSIGHTPROGRAM/Data/detection/detection/train/'


def data_preparation(root_dir, splitting_index=200):
    """
    Split the dataset in training and evaluation

    :param root_dir: path to folder containing subfolders images and masks
    :param splitting_index: determine the splitting between train and test data.
                            The train dataset is up to the index. The remaining part go into the test dataset
    :return:
    """
    # Data preparing code
    print("Preparation data")

    dataset = AppleDataset(root_dir, get_transform(train=True))
    #dataset_eval = AppleDataset(root_dir, get_transform(train=False))
    dataset_test = AppleDataset(root_dir, get_transform(train=False))

    # split the dataset in train and test set 70% train, 30% test
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()

    dataset = torch.utils.data.Subset(dataset, indices[:-splitting_index])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-splitting_index:])
    return dataset, dataset_test


def dataloader(dataset, dataset_test):
    # define training and validation data loaders
    print("Creating data loaders")
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    ## remember:check num_workers=4, in prediction it uses num_workers=1!!!
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    return data_loader, data_loader_test
