import numpy as np
from skimage.util.shape import view_as_windows
from scipy.ndimage import zoom
from scipy.ndimage.filters import gaussian_filter1d
import spectrogram as sp
from scipy.io import wavfile
import pyximport; pyximport.install()
import nms as nms

import theano
import lasagne
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers import Pool2DLayer as PoolLayer
from lasagne.layers import DenseLayer


class NeuralNet:

    def __init__(self, params_):
        self.params = params_
        self.network = None

    def train(self, positions, class_labels, files, durations, classes):
        feats = []
        labs = []
        for ii, file_name in enumerate(files):
            print file_name
            if positions[ii].shape[0] > 0:
                local_feats = self.create_or_load_features(file_name)

                # convert time in file to integer
                positions_ratio = positions[ii] / durations[ii]
                train_inds = (positions_ratio*float(local_feats.shape[0])).astype('int')
                feats.append(local_feats[train_inds, :, :, :])
                labs.append(class_labels[ii]*classes[ii])

       # flatten list of lists and set to correct output size
        features = np.vstack(feats)
        labels = np.vstack(labs).astype(np.uint8)[:,0] 
        # train network
        input_var = theano.tensor.tensor4('inputs')
        target_var = theano.tensor.ivector('targets')
        self.network = build_cnn(features.shape[2:], input_var, self.params.net_type)
        prediction = lasagne.layers.get_output(self.network['prob'])
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network['prob'], trainable=True)
        updates = lasagne.updates.nesterov_momentum(
                loss, params, learning_rate=self.params.learn_rate, momentum=self.params.moment)
        train_fn = theano.function([input_var, target_var], loss, updates=updates)

        # Training loop
        for epoch in range(self.params.num_epochs):
            for batch in iterate_minibatches(features, labels, self.params.batchsize, shuffle=True):
                inputs, targets = batch
                train_fn(inputs, targets)

        # test function           
        pred = lasagne.layers.get_output(self.network['prob'], deterministic=True)
        self.test_fn = theano.function([input_var], pred)

    def test(self, file_name=None, file_duration=None, audio_samples=None, sampling_rate=None):

        # compute features and perform classification
        features = self.create_or_load_features(file_name, audio_samples, sampling_rate)
    
        # predictions for max classes
        y_predictions = self.test_fn(features)
    

        # trying to get rid of rows with 0 highest
        #call_predictions  = y_predictions[np.argmax(y_predictions, axis=1) != 0][:]
        #print 'no 0 highest', call_predictions.shape
        call_predictions = y_predictions
        call_predictions = call_predictions[:,1:]
        #print 'no 0', call_predictions.shape
        
        high_preds = np.array([np.max(x) for x in call_predictions])[:, np.newaxis]
        classes = np.array([np.argmax(x)+1 for x in call_predictions])[:, np.newaxis]
        #print 'high_preds', high_preds.shape
        #print 'classes', classes.shape

    
        # smooth the output prediction
        if self.params.smooth_op_prediction:
            high_preds = gaussian_filter1d(high_preds, self.params.smooth_op_prediction_sigma, axis=0)
        
        #print 'after filter', high_preds.shape

        # perform non max suppression
        pos, prob, classes = nms.nms_1d(high_preds[:,0].astype(np.float), classes,self.params.nms_win_size, file_duration)
        #print 'pos', pos.shape
        #print 'prob', prob.shape
        #print 'classes nms', classes.shape
        return pos, prob, high_preds, classes

    def create_or_load_features(self, file_name=None, audio_samples=None, sampling_rate=None):
        """
        Does 1 of 3 possible things
        1) computes feature from audio samples directly
        2) loads feature from disk OR
        3) computes features from file name
        """

        if file_name is None:
            features = compute_features(audio_samples, sampling_rate, self.params)
        else:
            if self.params.load_features_from_file:
                features = np.load(self.params.feature_dir + file_name + '.npy')
            else:
                path = file_name
                if not '/storage' in file_name:
                    path = self.params.audio_dir + file_name
                sampling_rate, audio_samples = wavfile.read(path + '.wav')
                features = compute_features(audio_samples, sampling_rate, self.params)

        return features

    def save_features(self, files):
        for file_name in files:
            sampling_rate, audio_samples = wavfile.read(self.params.audio_dir + file_name + '.wav')
            features = compute_features(audio_samples, sampling_rate, self.params)
            np.save(self.params.feature_dir + file_name, features)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    # Note: this should not be used for testing as it creats even sized
    # minibatches so will skip some data
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]

def build_cnn(ip_size, input_var, net_type):
    if net_type == 'big':
        net = network_big(ip_size, input_var)
    elif net_type == 'small':
        net = network_sm(ip_size, input_var)
    else:
        print 'Error: network not defined'
    return net

def network_big(ip_size, input_var):
    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, 1, ip_size[0], ip_size[1]), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'], 32, 3, pad=1)
    net['pool1'] = PoolLayer(net['conv1'], 2)
    net['conv2'] = ConvLayer(net['pool1'], 32, 3, pad=1)
    net['pool2'] = PoolLayer(net['conv2'], 2)
    net['conv3'] = ConvLayer(net['pool2'], 32, 3, pad=1)
    net['pool3'] = PoolLayer(net['conv3'], 2)
    net['fc1']  = DenseLayer(lasagne.layers.dropout(net['pool3'], p=0.5), num_units=256, nonlinearity=lasagne.nonlinearities.rectify)
    net['prob'] = DenseLayer(lasagne.layers.dropout(net['fc1'], p=0.5), num_units=8, nonlinearity=lasagne.nonlinearities.softmax)
    return net

def network_sm(ip_size, input_var):
    net = {}
    net['input'] = lasagne.layers.InputLayer(shape=(None, 1, ip_size[0], ip_size[1]), input_var=input_var)
    net['conv1'] = ConvLayer(net['input'], 16, 3, pad=0)
    net['pool1'] = PoolLayer(net['conv1'], 2)
    net['conv2'] = ConvLayer(net['pool1'], 16, 3, pad=0)
    net['pool2'] = PoolLayer(net['conv2'], 2)
    net['fc1']  = DenseLayer(lasagne.layers.dropout(net['pool2'], p=0.5), num_units=64, nonlinearity=lasagne.nonlinearities.rectify)
    net['prob'] = DenseLayer(lasagne.layers.dropout(net['fc1'], p=0.5), num_units=2, nonlinearity=lasagne.nonlinearities.softmax)
    return net

def compute_features(audio_samples, sampling_rate, params):
    """
    Computes overlapping windows of spectrogram as input for CNN.
    """

    # load audio and create spectrogram
    spectrogram = sp.gen_spectrogram(audio_samples, sampling_rate, params.fft_win_length, params.fft_overlap,
                                     crop_spec=params.crop_spec, max_freq=params.max_freq, min_freq=params.min_freq)
    spectrogram = sp.process_spectrogram(spectrogram, denoise_spec=params.denoise, mean_log_mag=params.mean_log_mag, smooth_spec=params.smooth_spec)

    # extract windows
    spec_win = view_as_windows(spectrogram, (spectrogram.shape[0], params.window_width))[0]
    spec_win = zoom(spec_win, (1, 0.5, 0.5), order=1)
    spec_width = spectrogram.shape[1]

    # make the correct size for CNN
    features = np.zeros((spec_width, 1, spec_win.shape[1], spec_win.shape[2]), dtype=np.float32)
    features[:spec_win.shape[0], 0, :, :] = spec_win

    return features
