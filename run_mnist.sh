#!/bin/bash

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_mnist_fulluRNN512_lr0-0001_lrng0-000001_permuted_patience5_natGradRMS.yaml

THEANO_FLAGS="device=gpu0" python2.7 mnist.py -c config_mnist_restricteduRNNfast_lr0-0001_permuted_patience5.yaml

