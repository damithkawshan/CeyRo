'''
Evaluation script for the CeyRo Traffic Sign and Traffic Light Dataset

gt_dir should contain the ground truth xml files and pred_dir should contain prediction xml files respectively.
The file system should follow the following order.
home_directory/
|___ gt_dir/
|      |___001.xml
|      |___002.xml
|      |___ ....
|___ pred_dir/
       |___001.xml
       |___002.xml
       |___ ....
'''

from shapely.geometry import Polygon
from tabulate import tabulate
from os import listdir

import argparse

class_dict = {'DWS':0, 'MNS':0, 'PHS':0, 'PRS':0, 'SLS':0, 'OSD':0, 'APR':0, 'TLS':0}

def get_bbox_arr(fn):
    
    arr_shape = []
    
    with open(fn) as fo:
        tree = ET.parse(fn)
        root = tree.getroot()

        for child in root:
            
            if(child.tag == 'object'):
                arr_bboxes = {'label':None,"points":None}
                for grandChild in child:
                    if(grandChild.tag == 'bndbox'):
                        for coor in grandChild:
                            if(coor.tag == 'xmin'): x_1 = int(coor.text)
                            if(coor.tag == 'ymin'): y_1 = int(coor.text)
                            if(coor.tag == 'xmax'): x_2 = int(coor.text)
                            if(coor.tag == 'ymax'): y_2 = int(coor.text)

                        bbox = [[x_1,y_1],[x_1,y_2],[x_2,y_2],[x_2,y_1]]
                        

                    if(grandChild.tag == 'name'):
                        label = grandChild.text

                arr_bboxes['label'] = label
                arr_bboxes['points'] = bbox
                arr_shape.append(arr_bboxes)
    
    return arr_shape


def get_IoU(pol_1, pol_2):

    # Define each polygon
    polygon1_shape = Polygon(pol_1)
    polygon2_shape = Polygon(pol_2)

    if ~(polygon1_shape.is_valid):polygon1_shape = polygon1_shape.buffer(0)
    if ~(polygon2_shape.is_valid):polygon2_shape = polygon2_shape.buffer(0)

    # Calculate intersection and union, and return IoU
    polygon_intersection    = polygon1_shape.intersection(polygon2_shape).area
    polygon_union           = polygon1_shape.area + polygon2_shape.area - polygon_intersection

    return polygon_intersection / polygon_union

def match_gt_with_pred(gt_bboxes, pred_bboxes, iou_threshold):

    candidate_dict_gt  = {}  
    assigned_predictions = []

    # Iterate over ground truth
    for idx_gt, gt_itm in enumerate(gt_bboxes):
        pts_gt       = gt_itm['points']
        label_gt     = gt_itm['label']
        gt_candidate = {'label_pred':None, 'iou':0}
        assigned_prediction = None

        # Iterate over predictions
        for idx_pred, pred_itm in enumerate(pred_bboxes):
            pts_pred   = pred_itm['points']
            label_pred = pred_itm['label']
            iou        = get_IoU(pts_pred, pts_gt)
            
            # Match gt with predicitons
            if (iou > iou_threshold) and (gt_candidate['iou'] < iou) and (label_gt == label_pred) and str(idx_pred) not in assigned_predictions:
                gt_candidate['label_pred'] = label_pred + '*' + str(idx_pred)
                gt_candidate['iou']        = iou
                assigned_prediction        = str(idx_pred)
    
        if assigned_prediction is not None:
            assigned_predictions.append(assigned_prediction)

        candidate_dict_gt[label_gt + '*' + str(idx_gt)] = gt_candidate
    
    return candidate_dict_gt

def eval_detections(gt_dir, pred_dir, iou_threshold = 0.3):

    gt_xml_count   = len([f for f in listdir(gt_dir) if f.endswith('.xml')])
    pred_xml_count = len([f for f in listdir(pred_dir) if f.endswith('.xml')])
    
    assert gt_xml_count == pred_xml_count, "Ground truth xml file count does not match with prediction xml file count"

    print("Evaluating road marking detection performance on " + str(gt_xml_count) + " files")
    print()

    classwise_results    = [['Class', 'Precision', 'Recall', 'F1_Score']]

    filenames = [f for f in listdir(gt_dir) if f.endswith('.xml')]

    sigma_tp = 0
    sigma_fp = 0
    sigma_fn = 0

    tp_class_dict   = class_dict.copy()
    gt_class_dict   = class_dict.copy()
    pred_class_dict = class_dict.copy()


    # Iterate over each file
    for file in filenames:
        #Load ground truth xml file
        gt_bboxes   = get_bbox_arr((gt_dir + '/' + file) )
        # Load pred xml file
        pred_bboxes = get_bbox_arr((pred_dir + '/' + file))

        for bbox in gt_bboxes:
            gt_class_dict[bbox['label']]  += 1

        for bbox in pred_bboxes:
            pred_class_dict[bbox['label']]  += 1

        tp_gt = 0

        candidate_dict_gt = match_gt_with_pred(gt_bboxes, pred_bboxes, iou_threshold)

        for idx, lab in enumerate(candidate_dict_gt):
            label    = lab.split('*')[0]
            pred_lab = candidate_dict_gt[lab]['label_pred']

            if pred_lab != None:
                tp_gt                      += 1
                tp_class_dict[label]       += 1

        tp = tp_gt
        fp = len(pred_bboxes) - tp
        fn = len(gt_bboxes) - tp

        sigma_tp += tp
        sigma_fp += fp
        sigma_fn += fn

    # Calculate precision, recall and F1 for the whole dataset
    if (sigma_tp + sigma_fp) != 0:
        precision = sigma_tp / (sigma_tp + sigma_fp)
    else:
        precision = 0

    if (sigma_tp + sigma_fn) != 0:
        recall = sigma_tp / (sigma_tp + sigma_fn)
    else:
        recall = 0

    if (precision + recall) != 0:
        F1_score = (2 * precision * recall) / (precision + recall)
    else:
        F1_score = 0

    class_F1_scores_list = []

    # Calculate class-wise performance metrics
    for label in tp_class_dict:
        l_tp = tp_class_dict[label]
        l_fp = pred_class_dict[label] - l_tp
        l_fn = gt_class_dict[label] - l_tp

        if (l_tp + l_fp) != 0:
            l_precision = l_tp / (l_tp + l_fp)
        else:
            l_precision = 0

        if (l_tp + l_fn) != 0:
            l_recall = l_tp / (l_tp + l_fn)
        else:
            l_recall = 0

        if (l_precision + l_recall) != 0:
            l_F1_score = (2 * l_precision * l_recall) / (l_precision + l_recall)
        else:
            l_F1_score = 0

        classwise_results.append([label, round(l_precision, 4), round(l_recall, 4), round(l_F1_score, 4)])

        class_F1_scores_list.append(l_F1_score) 

    macro_F1_score = sum(class_F1_scores_list) / len(class_F1_scores_list)

    print('Class-wise road marking detection results')
    print(tabulate(classwise_results, headers='firstrow', tablefmt='grid'))
    print()

    print("Overall Precision : " + str(round(precision, 4)))
    print("Overall Recall    : " + str(round(recall, 4)))
    print("Overall F1-Score  : " + str(round(F1_score, 4)))
    print("Macro F1-Score    : " + str(round(macro_F1_score, 4)))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', type = str, help = 'Filepath containing ground truth xml files')
    parser.add_argument('--pred_dir', type = str, help = 'Filepath containing prediction xml files')
    parser.add_argument('--iou_threshold', type = float, default = 0.3, help = 'IoU threshold to count a prediction as a true positive')
    opt = parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    eval_detections(opt.gt_dir, opt.pred_dir, opt.iou_threshold)
