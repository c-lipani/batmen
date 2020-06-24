import numpy as np
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import evaluate as evl
import create_results as res
from data_set_params import DataSetParams
import classifier as clss
import pandas as pd
import cPickle as pickle


def read_baseline_res(baseline_file_name, test_files):
    da = pd.read_csv(baseline_file_name)
    pos = []
    prob = []
    for ff in test_files:
        rr = da[da['Filename'] == ff]
        inds = np.argsort(rr.TimeInFile.values)
        pos.append(rr.TimeInFile.values[inds])
        prob.append(rr.Quality.values[inds][..., np.newaxis])
    return pos, prob


if __name__ == '__main__':
    """
    This compares several different algorithms for bat echolocation detection.

    The results can vary by a few percent from run to run. If you don't want to
    run a specific model or baseline comment it out.
    """

    test_set = 'batmen'
    data_set = 'data/train_test_split/batmen.npz'
    raw_audio_dir = 'data/wav/'
    base_line_dir = 'data/baselines/'
    result_dir = 'results/'
    model_dir = 'data/models/'
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    print 'test set:', test_set
    plt.close('all')

    # train and test_pos are in units of seconds
    loaded_data_tr = np.load(data_set, allow_pickle=True)
    train_pos = loaded_data_tr['train_pos']
    train_files = loaded_data_tr['train_files']
    train_durations = loaded_data_tr['train_durations']
    test_pos = loaded_data_tr['test_pos']
    test_files = loaded_data_tr['test_files']
    test_durations = loaded_data_tr['test_durations']

    # TEST 
    train_classes = loaded_data_tr['train_class']
    test_classes = loaded_data_tr['test_class']
    
    # load parameters
    params = DataSetParams()
    params.audio_dir = raw_audio_dir

    #
    # CNN
    print '\ncnn'
    params.classification_model = 'cnn'
    model = clss.Classifier(params)
    
    # train
    model.train(train_files, train_pos, train_durations, train_classes)

    # test all
    nms_pos, nms_prob, class_  = model.test_batch(test_files, test_pos, test_durations, False, 'spectrograms/')

    # save CNN model to file
    pickle.dump(model, open(model_dir + 'test_set_' + test_set + '.mod', 'wb'))
    print 'cnn done'
