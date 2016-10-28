#!/usr/bin/python
'''Pixel-by-pixel MNIST using a unitary RNN (uRNN)
'''

from __future__ import print_function

import os,sys,getopt
import yaml
import cPickle

import numpy as np
np.random.seed(314159)  # for reproducibility

import keras.callbacks
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation,LSTM,TimeDistributed
from keras.layers import SimpleRNN
from keras.initializations import normal, identity
from keras.optimizers import RMSprop
from keras.utils import np_utils
from custom_layers import uRNN,complex_RNN_wrapper
from custom_optimizers import RMSprop_and_natGrad


class LossHistory(keras.callbacks.Callback):
    def __init__(self, histfile):
        self.histfile=histfile
    
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_acc  = []
        self.val_loss   = []
        self.val_acc    = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.val_loss.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        cPickle.dump({'train_loss' : self.train_loss, 'train_acc' : self.train_acc, 'val_loss': self.val_loss, 'val_acc' : self.val_acc}, open(self.histfile, 'wb'))     


def main(argv):
    config={'learning_rate' : 1e-4,
            'learning_rate_natGrad' : None,
            'clipnorm' : 1.0,
            'batch_size' : 32,
            'nb_epochs' : 200,
            'patience' : 3,
            'hidden_units' : 100,
            'model_impl' : 'complex_RNN',
            'unitary_impl' : 'ASB2016',
            'histfile' : 'exp/history_mnist_default',
            'savefile' : 'exp/model_mnist_default.hdf5',
            'savefile_init' : None}

    configfile = ''
    helpstring = 'mnist_urnn.py -c <config YAML file>'
    try:
        opts, args = getopt.getopt(argv,"hc:",["config="])
    except getopt.GetoptError:
        print(helpstring)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(helpstring)
            yamlstring = yaml.dump(config,default_flow_style=False,explicit_start=True)
            print("YAML configuration file format:")
            print("")
            print("%YAML 1.2")
            print(yamlstring)
            sys.exit()
        elif opt in ("-c","--config"):
            configfile=arg
    print("Config file is %s" % configfile)
    if os.path.exists(configfile):
        f = open(configfile)
        user_config = yaml.load(f.read())
        config.update(user_config)

    print("Printing configuration:")
    for key,value in config.iteritems():
        print("  ",key,": ",value)

    nb_classes = 10
    
    learning_rate = config['learning_rate']
    if ('learning_rate_natGrad' in config) and (config['learning_rate_natGrad'] is not None):
        learning_rate_natGrad = config['learning_rate_natGrad']
    else:
        learning_rate_natGrad = learning_rate
    clipnorm = config['clipnorm']
    batch_size = config['batch_size']
    nb_epochs = config['nb_epochs']
    hidden_units = config['hidden_units']
    # ASB2016 uRNN has 32N+10 parameters
    # full uRNN has N^2+25N+10 parameters

    #model_impl='uRNN_keras'
    #model_impl='complex_RNN'
    model_impl=config['model_impl']
    unitary_impl=config['unitary_impl']

    histfile=config['histfile']
    savefile=config['savefile']

    # the data, shuffled and split between train, validation, and test sets
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_valid = X_train[:5000,:,:]
    y_valid = y_train[:5000]
    X_train = X_train[5000:,:,:]
    y_train = y_train[5000:]

    X_train = X_train.reshape(X_train.shape[0], -1, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1, 1)
    X_test = X_test.reshape(X_test.shape[0], -1, 1)
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_valid /= 255
    X_test /= 255
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_valid.shape[0], 'validation samples')
    print(X_test.shape[0], 'test samples')

    if ('flag_permute' in config) and config['flag_permute']:
        print("Applying permutation to MNIST pixels")
        rng_permute = np.random.RandomState(92916)
        idx_permute = rng_permute.permutation(784)
        X_train=X_train[:,idx_permute]
        X_valid=X_valid[:,idx_permute]
        X_test =X_test[:,idx_permute]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('Building model with implementation %s...' % model_impl)
    if (model_impl=='uRNN_keras'):
        #unitary_init='svd'
        unitary_init='ASB2016'

        unitary_impl='ASB2016'
        #unitary_impl='full'
        #unitary_impl='full_natGrad'
        
        epsilon=1e-5
        
        model = Sequential()
        model.add(uRNN(output_dim=hidden_units,
                            inner_init=unitary_init,
                            unitary_impl=unitary_impl,
                            input_shape=X_train.shape[1:],
                            consume_less='cpu',
                            epsilon=epsilon))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))
    elif (model_impl=='complex_RNN'):
        model = Sequential()
        model.add(complex_RNN_wrapper(output_dim=nb_classes,
                              hidden_dim=hidden_units,
                              unitary_impl=unitary_impl,
                              input_shape=X_train.shape[1:]))
        model.add(Activation('softmax'))
    elif (model_impl=='LSTM'):
        model = Sequential()
        model.add(LSTM(hidden_units,
                       return_sequences=False,
                       input_shape=X_train.shape[1:]))
        model.add(Dense(nb_classes))
        model.add(Activation('softmax'))

    rmsprop = RMSprop_and_natGrad(lr=learning_rate,clipnorm=clipnorm,lr_natGrad=learning_rate_natGrad)
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    history=LossHistory(histfile)
    checkpointer = keras.callbacks.ModelCheckpoint(filepath=savefile, verbose=1, save_best_only=True)
    earlystopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=config['patience'], verbose=1, mode='auto') 

    if not (config['savefile_init'] is None):
        print("Loading weights from file %s" % config['savefile_init'])
        model.load_weights(config['savefile_init'])
        losses = model.test_on_batch(X_valid,Y_valid)
        print("On validation set, loaded model achieves loss %f and acc %f"%(losses[0],losses[1]))

    #make sure the experiment directory to hold results exists
    if not os.path.exists('exp'):
        os.makedirs('exp')

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
              verbose=1, validation_data=(X_valid, Y_valid),callbacks=[history,checkpointer,earlystopping])

    scores = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    # add test scores to history
    history_load=cPickle.load(open(histfile,'rb'))
    history_load.update({'test_loss' : scores[0], 'test_acc' : scores[1]})
    cPickle.dump(history_load, open(histfile, 'wb'))     

if __name__ == "__main__":
    main(sys.argv[1:])
