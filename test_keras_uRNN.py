import os,sys,getopt,yaml
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

def main(argv):
    config={
            'hidden_units' : 100,
            'model_impl' : 'uRNN_keras',
            'unitary_init' : 'ASB2016',
            'unitary_impl' : 'full_natGradRMS',
            'consume_less' : 'mem',
            'batch_size' : 50,
            'lr' : 1e-4,
            'lr_natGrad' : 1e-6,
            'nb_epoch' : 3,
            'clipnorm' : 0.,
            'tests' : ['predict1','predict2','train1','train2'],
            'ntrials_predict':1
            }

    configfile = ''
    helpstring = 'commandline_configfile -c <config YAML file>'
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
        print "  ",key,":",value
    print ""

    # write your program here
    
    #load MNIST data shuffled and split between train, validation, and test sets
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
    print 'X_train shape:',X_train.shape
    print X_train.shape[0], 'train samples'
    print X_valid.shape[0], 'validation samples'
    print X_test.shape[0], 'test samples'
    # convert class vectors to binary class matrices
    nb_classes=10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    #create a keras uRNN
    epsilon=1e-5
    model = Sequential()
    model.add(uRNN(output_dim=config['hidden_units'],
                        inner_init=config['unitary_init'],
                        unitary_impl=config['unitary_impl'],
                        input_shape=X_train.shape[1:],
                        consume_less=config['consume_less'],
                        epsilon=epsilon))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    #create a complex_RNN wrapper
    model2 = Sequential()
    model2.add(complex_RNN_wrapper(output_dim=nb_classes,
                          hidden_dim=config['hidden_units'],
                          unitary_impl=config['unitary_impl'],
                          input_shape=X_train.shape[1:]))
    model2.add(Activation('softmax'))

    #initialize Keras model with parameter values from complex_RNN_wrapper
    #model weights : [urnn_1_W, urnn_1_U, urnn_1_b, dense_1_W, dense_1_b]
    #model2 weights: [V, U, hidden_bias, out_bias, h_0, Waug_natGrad_unitaryAug]
    print "Initializing Keras uRNN with weights from complex_RNN_wrapper..."
    print ""
    # input transform V
    model.weights[0].set_value(model2.weights[0].get_value())
    # output transform U
    model.weights[-2].set_value(model2.weights[1].get_value())
    # hidden bias
    model.weights[2].set_value(model2.weights[2].get_value())
    # output bias
    model.weights[-1].set_value(model2.weights[3].get_value())
    # initial hidden state
    h02=model2.weights[4].get_value().flatten()
    #model2.weights[4].set_value(np.zeros_like(h02).astype(np.float32))
    model.weights[3].set_value(h02)
    # unitary recurrence matrix
    Waug=model2.weights[-1].get_value()
    """
    ReW=Waug[:Waug.shape[0]/2,:Waug.shape[1]/2]
    ImW=Waug[:Waug.shape[0]/2,Waug.shape[1]/2:]
    model.weights[1].set_value(np.concatenate( (ReW,ImW),axis=0 ))
    """
    model.weights[1].set_value(Waug)

    #now let's make sure we get the same output from the models
    if ('predict1' in config['tests']):
        for i in range(config['ntrials_predict']):
            Yest =model.predict_on_batch(X_valid)
    if ('predict2' in config['tests']):
        for i in range(config['ntrials_predict']):
            Yest2=model2.predict_on_batch(X_valid)
    if ('predict1' in config['tests']) and ('predict2' in config['tests']):
        print "Before training, NMSE between Keras uRNN and complex_RNN_wrapper is",np.mean((Yest-Yest2)**2)/np.mean(Yest2**2)
        print ""
    
    rmsprop = RMSprop_and_natGrad(lr=config['lr'],clipnorm=config['clipnorm'],lr_natGrad=config['lr_natGrad'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])
    model2.compile(loss='categorical_crossentropy',
                  optimizer=rmsprop,
                  metrics=['accuracy'])

    #model.train_on_batch(X_train[:config['batch_size']],Y_train[:config['batch_size']])
    #model2.train_on_batch(X_train[:config['batch_size']],Y_train[:config['batch_size']])


    if ('train1' in config['tests']):
        model.fit(X_train[:10*config['batch_size']],Y_train[:10*config['batch_size']],batch_size=config['batch_size'],nb_epoch=config['nb_epoch'],verbose=1)
        Yest =model.predict_on_batch(X_valid)
    if ('train2' in config['tests']):
        model2.fit(X_train[:10*config['batch_size']],Y_train[:10*config['batch_size']],batch_size=config['batch_size'],nb_epoch=config['nb_epoch'],verbose=1)
        Yest2=model2.predict_on_batch(X_valid)
    if ('train1' in config['tests']) and ('train2' in config['tests']):
        print "After training, NMSE between Keras uRNN and complex_RNN_wrapper is",np.mean((Yest-Yest2)**2)/np.mean(Yest2**2)
        print ""

if __name__ == "__main__":
    main(sys.argv[1:])

