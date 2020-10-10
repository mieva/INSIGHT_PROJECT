import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt

# Upload the predictions in a pandas df
def pred_to_df(output_file):
  # Read the file
  df_predictions = pd.read_csv(output_file)
  # Add column names
  #df_predictions.columns=['img_name','p_left', 'p_top', 'p_width', 'p_height', 'score']
  df_predictions.columns=['img_name','p_x1', 'p_y1', 'p_x2', 'p_y2', 'score']
  # Convert boxes coordinate to the right type
  df_predictions['p_x1'] = df_predictions.p_x1.astype(float)
  df_predictions['p_y1'] = df_predictions.p_y1.astype(float)
  df_predictions['p_x2'] = df_predictions.p_x2.astype(float)
  df_predictions['p_y2'] = df_predictions.p_y2.astype(float)

  return df_predictions

def gt_to_df(output_file):
  # Read the file
  df = pd.read_csv(output_file)
  df.columns=['img_name','gt_x1', 'gt_y1', 'gt_x2', 'gt_y2']
  # Convert boxes coordinate to the right type
  df['gt_x1'] = df.gt_x1.astype(float)
  df['gt_y1'] = df.gt_y1.astype(float)
  df['gt_x2'] = df.gt_x2.astype(float)
  df['gt_y2'] = df.gt_y2.astype(float)

  return df

# Function to calculate the
def iou(xbl_1, ybl_1, xtr_1, ytr_1, xbl_2, ybl_2, xtr_2, ytr_2):
    """
    :param xbl_1: x coord of bottom left corner of first box
    :param ybl_1: y coord of bottom left corner of first box
    :param xtr_1: x coord of top right corner of first box
    :param ytr_1: y coord of top right corner of first box

    :param xbl_2: x coord of bottom left corner of second box
    :param ybl_2: y coord of bottom left corner of second box
    :param xtr_2: x coord of top right corner of second box
    :param ytr_2: y coord of top right corner of second box
    """
    if np.isnan(xbl_2):
        return 0
    box_1 = Polygon([[xbl_1, ybl_1], [xbl_1, ytr_1], [xtr_1, ytr_1], [xtr_1, ybl_1]])
    box_2 = Polygon([[xbl_2, ybl_2], [xbl_2, ytr_2], [xtr_2, ytr_2], [xtr_2, ybl_2]])
    return box_1.intersection(box_2).area / box_1.union(box_2).area

# Function to get df_metrics
# Function to get df_metrics for both models
def get_df_metrics(df_gt, df_predictions, score_cut=0.9):

    df = pd.merge(df_gt, df_predictions, how='left', on='img_name')

    # Calculate iou for each image and for each gt boxes
    df['iou'] = df[['gt_x1', 'gt_y1', 'gt_x2', 'gt_y2', 'p_x1', 'p_y1', 'p_x2', 'p_y2']]. \
        apply(lambda x: iou(*x), axis=1)
    # img box_gt box_p iou

    # Take the best candidate
    df_iou = df.groupby(['img_name','gt_x1', 'gt_y1', 'gt_x2', 'gt_y2'])['iou'].max().reset_index()
    # img box_gt best_iou

    # Apply the cut at 0.5 and create the metrics df
    df_metrics = \
    pd.merge(pd.merge(
        df_predictions[df_predictions['score']>score_cut]. \
            groupby('img_name')['score'].count().reset_index().rename(columns={"score": "p_num_boxes"}),
        df_iou[df_iou['iou'] > 0.5].groupby('img_name')['iou'].count().reset_index().rename(columns={"iou": "true_pos"}),
        on='img_name', how='inner'),
             df_gt.groupby('img_name')['gt_x1'].count().reset_index().rename(columns={"gt_x1": "t_num_boxes"}),
             on='img_name', how='inner'
             )
    return df_metrics

# Function to calculate precision and recall
def get_prec_recal(df):
  df['precision'] = df['true_pos']/df['p_num_boxes']
  df['recall'] = df['true_pos']/df['t_num_boxes']
  df['fscore'] = 2*df['precision']*df['recall']/(df['precision'] + df['recall'])
  # Fill nan with zero
  df.loc[:, ['precision', 'recall', 'fscore']] = df[['precision', 'recall', 'fscore']].fillna(0)
  return df

# Function to evaluate the overall Precision and Recall
def get_overall_performance(df):
    precision = (df['true_pos'].sum()/df['p_num_boxes'].sum())
    recall = (df['true_pos'].sum()/df['t_num_boxes'].sum())
    fscore = 2*precision*recall/(precision + recall)
    return precision, recall, fscore


def make_evaluation_on_dataset(gt_file, prediction_file):
    """
    Function to evaluate model performances
    :param gt_file: Groung_truth file with bbx
    :param prediction_file: Predicition file with bbx
    :return:
    """
    # Ground truth
    df_gt = gt_to_df(gt_file)
    # Predictions
    df_predictions = pred_to_df(prediction_file)

    # Create the metrics data frame
    df_metrics = get_df_metrics(df_gt, df_predictions)

    # Precision and recall evaluation for rcnn
    df_metrics = get_prec_recal(df_metrics)

    # Plot the overall recall and precision
    fig, axes = plt.subplots(sharex=True, figsize=(9, 12), nrows=3, ncols=1)
    df_metrics['precision'].hist(bins=15, color='darkblue',
                                 range=(0, 1), alpha=0.5, log=False,
                                 ax=axes[0], fill=True, histtype='step', hatch='\\')
    df_metrics['recall'].hist(bins=15, color='darkred',
                              range=(0, 1), alpha=0.5, log=False,
                              ax=axes[1], fill=True, histtype='step', hatch='\\')
    df_metrics['fscore'].hist(bins=15, color='darkgreen',
                              range=(0, 1), alpha=0.5, log=False,
                              ax=axes[2], fill=True, histtype='step', hatch='\\')
    axes[0].set_title('Precision')
    axes[0].set(xlabel='Precision')
    axes[1].set_title('Recall')
    axes[1].set(xlabel='Recall')
    axes[2].set_title('F1_score')
    axes[2].set(xlabel='F1_score')

    plt.savefig('/content/drive/My Drive/INSIGHTPROGRAM/MODEL_OUTPUT/FastRCNN_metrics.png')

    # Evaluation for FastRCNN
    overall_precision, overall_recall, overall_fscore = get_overall_performance(df_metrics)
    print(f'The overall precision is {overall_precision:.3f}')
    print(f'The overall recall is {overall_recall:.3f}.')
    print(f'The overall F1_score is {overall_fscore:.3f}.')