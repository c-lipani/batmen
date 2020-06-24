import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from data_set_params import DataSetParams
import create_results as res
import evaluate as evl

def remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o,class_o, durations, win_size):
    # this filters out predictions and gt that are close to the end
    # this is a bit messy because of the shapes of gt_pos_o
    nms_pos = []
    nms_prob = []
    gt_pos = []
    class_ = []
    for ii in range(len(nms_pos_o)):
        valid_time = durations[ii] - win_size
        gt_cur = gt_pos_o[ii]
        if gt_cur.shape[0] > 0:
            gt_pos.append(gt_cur[:, 0][gt_cur[:, 0] < valid_time][..., np.newaxis])
        else:
            gt_pos.append(gt_cur)

        valid_preds = nms_pos_o[ii] < valid_time
        nms_pos.append(nms_pos_o[ii][valid_preds])
        class_.append(class_o[ii][valid_preds])
        nms_prob.append(nms_prob_o[ii][valid_preds, 0][..., np.newaxis])
    return nms_pos, nms_prob, gt_pos, class_



def conf_matrix(nms_pos_o, nms_prob_o, gt_pos_o,class_o, classes, durations, detection_overlap, win_size, remove_eof=True):
    if remove_eof:
        # filter out the detections in both ground truth and predictions that are too
        # close to the end of the file - dont count them during eval
        nms_pos, nms_prob, gt_pos, class_ = remove_end_preds(nms_pos_o, nms_prob_o, gt_pos_o, class_o,durations, win_size)
    else:
        nms_pos = nms_pos_o
        nms_prob = nms_prob_o
        gt_pos = gt_pos_o
        class_ = class_o

    # loop through each file

    true_pos = []  # correctly predicts the ground truth
    false_pos = []  # says there is a detection but isn't
    conf_matrix = np.zeros((7,7), dtype=int)

    for ii in range(len(nms_pos)):
        num_preds = nms_pos[ii].shape[0]
        cur_class = classes[ii]

        if num_preds > 0:  # check to make sure it contains something
            num_gt = gt_pos[ii].shape[0]

            # for each set of predictions label them as true positive or false positive (i.e. 1-tp)
            tp = np.zeros(num_preds)
            distance_to_gt = np.abs(gt_pos[ii].ravel()-nms_pos[ii].ravel()[:, np.newaxis])
            within_overlap = (distance_to_gt <= detection_overlap)
            
            # remove duplicate detections - assign to valid detection with highest prob
            for jj in range(num_gt):
                inds = np.where(within_overlap[:, jj])[0] # get the indices of all valid predictions
                if inds.shape[0] > 0:
                    max_prob = np.argmax(nms_prob[ii][inds])
                    selected_pred = inds[max_prob]
                    within_overlap[selected_pred, :] = False
                    tp[selected_pred] = 1  # set as true positives
            true_pos.append(tp)
            false_pos.append(1 - tp)
            
            for i in range(len(class_[ii])):
                conf_matrix[cur_class-1][class_[ii][i]-1] = conf_matrix[cur_class-1][class_[ii][i]-1]+1

    print conf_matrix.sum()
    print '--------------'
    print 'Confusion matrix'
    print '--------------'
    print conf_matrix

    TP = np.diag(conf_matrix)
    FP = conf_matrix.sum(axis=0) - TP
    FN = conf_matrix.sum(axis=1) - TP
    TN = conf_matrix.sum() - (TP + FN + FP)
    AC = (TP+TN)/(TP+FP+FN+TN).astype(float)
    PRE = TP/(TP+FP).astype(float)
    REC = TP/(TP+FN).astype(float)
    
    print "sum", conf_matrix.sum()
    print TP
    print FP
    print FN
    print TN
    print AC
    print PRE
    print REC

    for i in range(conf_matrix.shape[0]):
        print '--------------'
        print 'Class', i+1
        print '--------------'
        print 'True Positive', TP[i]
        print 'False Positive', FP[i]
        print 'False Negative', FN[i]
        print 'True Negative',TN[i]
        print 'Accuracy', AC[i]
        print 'Precision', PRE[i]
        print 'Recall', REC[i]
        print
    
    print conf_matrix.sum()
    print '--------------'
    print 'Average'
    print '--------------'    
    print 'Average Accuracy', np.mean(AC)
    print 'Average Precision', np.mean(PRE)
    print 'Average Recall', np.mean(REC)


data_set = 'data/train_test_split/batmen.npz'
loaded_data_tr = np.load(data_set, allow_pickle=True)
test_pos = loaded_data_tr['test_pos']
test_files = loaded_data_tr['test_files']
test_durations = loaded_data_tr['test_durations']
test_classes = loaded_data_tr['test_class']
params = DataSetParams()

"""
nms_pos = np.load('pos3.npy')
nms_prob = np.load('prob3.npy')
class_ = np.load('classes3.npy')
"""
nms_pos = np.load('pos3.npy')
nms_prob = np.load('prob3.npy')
class_ = np.load('classes3.npy')

conf_matrix(nms_pos, nms_prob, test_pos ,class_, test_classes, test_durations, params.detection_overlap, params.window_size)
