import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


# Defying an istance for Resnet50
def get_instance_frcnn_model(num_classes, freeze=False):
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # freeze parameters from the most inner layers...
    # if freeze:
    # for param in model.parameters():
    #     param.requires_grad = False

    # replace the classifier with a new one, that has
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze:
        for name, param in model.named_parameters():
            if (name.split('.')[0] == 'roi_heads') or \
                (name.split('.')[0] == 'rpn') or \
                (name.split('.')[:2]) == ['backbone', 'fpn']:
                param.requires_grad = True
            else:
                param.requires_grad = False

    return model

## Defying an istance for mask Resnet50
def get_instance_segmentation_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model