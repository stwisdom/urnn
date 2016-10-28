# -*- coding: utf-8 -*-
import numpy as np

from keras import backend as K
from keras import activations, initializations, regularizers
from keras.engine import Layer, InputSpec
from keras.layers import Recurrent, time_distributed_dense
from keras.engine.topology import Layer

import theano
import theano.tensor as T

from fftconv import cufft, cuifft  

import models

def augLeft(ReIm,module=K):
    # return real-imaginary augmented matrix for left matrix multiplication
    N=ReIm.shape[0]/2
    Re=ReIm[:N]
    Im=ReIm[N:]
    return module.concatenate( \
            [ module.concatenate([Re,-Im],axis=1), \
              module.concatenate([Im, Re],axis=1)], axis=0)


def augRight(ReIm,module=K):
    # return real-imaginary augmented matrix for right matrix multiplication
    N=ReIm.shape[0]/2
    Re=ReIm[:N]
    Im=ReIm[N:]
    return module.concatenate( \
            [ module.concatenate([Re, Im],axis=1), \
              module.concatenate([-Im,Re],axis=1)], axis=0)


def build_swap_re_im(N):
    idx_re=np.arange(N)
    return np.concatenate([N+idx_re,idx_re],axis=0)


def do_fft(input, n_hidden):
    fft_input = K.reshape(input, (input.shape[0], 2, n_hidden))
    fft_input = fft_input.dimshuffle(0,2,1)
    fft_output = cufft(fft_input) / K.sqrt(n_hidden)
    fft_output = fft_output.dimshuffle(0,2,1)
    output = K.reshape(fft_output, (input.shape[0], 2*n_hidden))
    return output


def do_ifft(input, n_hidden):
    ifft_input = K.reshape(input, (input.shape[0], 2, n_hidden))
    ifft_input = ifft_input.dimshuffle(0,2,1)
    ifft_output = cuifft(ifft_input) / K.sqrt(n_hidden)
    ifft_output = ifft_output.dimshuffle(0,2,1)
    output = K.reshape(ifft_output, (input.shape[0], 2*n_hidden))
    return output


def times_diag(input, n_hidden, diag, swap_re_im):
    # input is a Ix2n_hidden matrix, where I is number
    # of training examples
    # diag is a n_hidden-dimensional real vector, which creates
    # the 2n_hidden x 2n_hidden complex diagonal matrix using
    # e.^{j.*diag}=cos(diag)+j.*sin(diag)
    d = K.concatenate([diag, -diag]) #d is 2n_hidden
    Re = K.cos(d).dimshuffle('x',0)
    Im = K.sin(d).dimshuffle('x',0)
    input_times_Re = input * Re
    input_times_Im = input * Im
    output = input_times_Re + input_times_Im[:, swap_re_im]
    return output


def vec_permutation(input, index_permute):
    return input[:, index_permute]


def Kouter(x1,x2):
    y=K.dot(K.expand_dims(x1,dim=-1),K.expand_dims(x2,dim=0))
    return y


def times_reflection(input, n_hidden, reflection):
    input_re = input[:, :n_hidden]
    input_im = input[:, n_hidden:]
    
    reflect_re = reflection[:n_hidden]
    reflect_im = reflection[n_hidden:]
    
    vstarv = (reflection**2).sum()
    
    input_re_reflect_re = K.dot(input_re, reflect_re)
    input_re_reflect_im = K.dot(input_re, reflect_im)
    input_im_reflect_re = K.dot(input_im, reflect_re)
    input_im_reflect_im = K.dot(input_im, reflect_im)
    
    a = Kouter(input_re_reflect_re - input_im_reflect_im, reflect_re)
    b = Kouter(input_re_reflect_im + input_im_reflect_re, reflect_im)
    c = Kouter(input_re_reflect_re - input_im_reflect_im, reflect_im)
    d = Kouter(input_re_reflect_im + input_im_reflect_re, reflect_re)
    
    output = input
    output = T.inc_subtensor(output[:, :n_hidden], - 2. / vstarv * (a + b))
    output = T.inc_subtensor(output[:, n_hidden:], - 2. / vstarv * (d - c))
    
    return output


def times_unitary_ASB2016(xaug,N,Uparams):
    theta=Uparams[0]
    reflection=Uparams[1]
    idxpermaug=Uparams[2]
    swap_re_im=build_swap_re_im(N)
    step1 = times_diag(xaug, N, theta[0,:], swap_re_im)
    step2 = do_fft(step1, N)
    step3 = times_reflection(step2, N, reflection[0,:])
    step4 = vec_permutation(step3, idxpermaug)
    step5 = times_diag(step4, N, theta[1,:], swap_re_im)
    step6 = do_ifft(step5, N)
    step7 = times_reflection(step6, N, reflection[1,:])
    step8 = times_diag(step7, N, theta[2,:], swap_re_im)
    yaug  = step8
    return yaug


def unitary_ASB2016_init(shape, name=None):
    assert shape[0]==shape[1]
    N=shape[1]
    
    theta = initializations.uniform((3,N),scale=np.pi,name='{}_theta'.format(name))
    reflection = initializations.glorot_uniform((2,2*N),name='{}_reflection'.format(name))
    idxperm = np.random.permutation(N)
    idxpermaug = np.concatenate((idxperm,N+idxperm))
    
    Iaug=augLeft(np.concatenate((np.eye(N),np.zeros((N,N))),axis=0),module=np).astype(np.float32)
    Uaug=times_unitary_ASB2016(Iaug,N,[theta,reflection,idxpermaug])

    return Uaug,theta,reflection,idxpermaug


def unitary_svd_init(shape, name=None):
    assert shape[0]==shape[1]
    
    Re=initializations.normal(shape,scale=1.0,name=name).get_value()
    Im=initializations.normal(shape,scale=1.0,name=name).get_value()
    X = Re+1j*Im
    [U,S,V]=np.linalg.svd(X)
    X = np.dot(U,V)
    ReX = np.real(X)
    ImX = np.imag(X)
    Xaug = np.concatenate([ReX,ImX],axis=0)
    return K.variable(Xaug,name=name)


class uRNN(Recurrent):
    '''Unitary RNN where the output is to be fed back to input, the
       hidden state is complex-valued, and the recurrence matrix U
       is unitary. Input transform is complex-valued.

    # Arguments
        output_dim: dimension of the complex-valued internal projections and the final output. Since hidden state of uRNN is complex-valued, self.output_dim will be equal to 2*output_dim. For a N-dimensional complex-valued hidden state, use output_dim=N.
        init: weight initialization function.
            Can be the name of an existing function (str),
            or a Theano function (see: [initializations](../initializations.md)).
        inner_init: initialization function of the inner cells.
            Options:
            'svd'    : samples random complex-valued Gaussian matrix
                               and makes it unitary by taking SVD and setting
                               all singular values to 1. 
                               Use with 'full' impl.s
            'ASB2016': uses parameterization of 
                               [Arjovsky,Shah,Bengio 2016].
                               Use with 'ASB2016' impl.
        activation: activation function.
            Only 'soft_thresh' supported for now
        unitary_impl: implementation of unitary recurrence matrix
            Options: 
            'ASB2016'     : uses parameterization of [Arjovsky,Shah,Bengio 2016]
            'full'        : uses full unitary matrix without unitary constraint
                            during optimization
            'full_natGrad': uses full unitary matrix with natural gradient step
                           (requires using <optimizer>_and_natGrad optimizer)
        input_type: either 'real' or 'complex', useful when stacking uRNNs
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the input weights matrices.
        U_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the recurrent weights matrices.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        dropout_W: float between 0 and 1. Fraction of the input units to drop for input gates.
        dropout_U: float between 0 and 1. Fraction of the input units to drop for recurrent connections.

    # References
        - [A Theoretically Grounded Application of Dropout in Recurrent Neural Networks](http://arxiv.org/abs/1512.05287)
        - [Unitary Evolution Recurrent Networks]()
        - [Full-Capacity Unitary Recurrent Neural Networks]()
    '''
    def __init__(self, output_dim,
                 init='glorot_uniform', 
                 inner_init='svd',
                 activation='soft_thresh',
                 unitary_impl='full_natGrad',
                 input_type='real',
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 dropout_W=0., dropout_U=0., 
                 epsilon=1e-5,
                 h0_mean=0.0,
                 **kwargs):
        idx_re = np.arange(output_dim)
        self.swap_re_im = build_swap_re_im(output_dim)
        self.output_dim = 2*output_dim #because the output will be complex-valued
        self.N = output_dim
        self.epsilon=epsilon
        self.h0_mean=h0_mean

        if (input_type=='real'):
            # W maps from real-valued inputs to complex-valued outputs
            self.init = initializations.get(init)
        elif (input_type=='complex'):
            # W maps from complex-valued inputs to complex-valued outputs
            print "Need to implement complex-valued uRNN inputs"
            raise NotImplementedError
        else:
            print "Input type of '%s' not supported" % input_type
            raise NotImplementedError

        if not ( (inner_init=='svd') or (inner_init=='ASB2016') ):
            print "Unitary recurrence initialization '%s' not supported" % inner_init
            raise NotImplementedError
        self.inner_init=inner_init

        if (activation=='soft_thresh'):
            # soft-threshold is [x/abs(x)]*relu(abs(x)+b)
            self.activation='soft_thresh'
        else:
            print "Activation '%s' not supported for unitary RNN" % activation
            raise NotImplementedError
        
        self.unitary_impl=unitary_impl
        if (self.unitary_impl=='ASB2016'):
            #always use ASB2016 init for ASB2016 impl
            self.inner_init = 'ASB2016'

        if (W_regularizer is not None) \
           or (U_regularizer is not None) \
           or (b_regularizer is not None) \
           or (dropout_W > 0.) \
           or (dropout_U > 0.):
            #self.W_regularizer = regularizers.get(W_regularizer)
            #self.U_regularizer = regularizers.get(U_regularizer)
            #self.b_regularizer = regularizers.get(b_regularizer)
            print "Regularizers and dropout not yet supported for unitary RNN"
            raise NotImplementedError

        self.dropout_W, self.dropout_U = dropout_W, dropout_U 
        if self.dropout_W or self.dropout_U:
            self.uses_learning_phase = True
        
        super(uRNN, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]
        input_dim = input_shape[2]
        self.input_dim = input_dim

        self.W = self.init((input_dim, self.output_dim),
                           name='{}_W'.format(self.name))
        #self.b = K.zeros((self.N,), name='{}_b'.format(self.name))
        self.b = initializations.uniform((self.N,),scale=0.01,name='{}_b'.format(self.name))
        self.baug=K.tile(self.b,[2])

        h0 = self.h0_mean+initializations.uniform((2*self.N,),scale=0.01).get_value()
        self.h0 = K.variable(h0,name='{}_h0'.format(self.name))

        if ('full' in self.unitary_impl):   
            # we're using a full unitary recurrence matrix
            
            if (self.inner_init=='svd'):
                # use SVD to initialize U
                self.U = unitary_svd_init((self.N, self.N),name='{}_U'.format(self.name))
            elif (self.inner_init=='ASB2016'):
                # use parameterization of [ASB2016] to initialize U
                Uaug,_,_,_ = unitary_ASB2016_init((self.N,self.N))
                Uaug=Uaug.eval()
                self.U=K.variable(np.concatenate((Uaug[:self.N,:self.N],Uaug[:self.N,self.N:]),axis=0),name='{}_U'.format(self.name))
                
            self.Uaug=augRight(self.U,module=K)

        elif (self.unitary_impl=='ASB2016'):
            # we're using the parameterization of [Arjovsky, Shah, Bengio 2016]
            self.Uaug,self.theta,self.reflection,_ = unitary_ASB2016_init((self.N, self.N),name=self.name)
        
        # set the trainable weights
        if ('full' in self.unitary_impl):
            self.trainable_weights = [self.W, self.U, self.b, self.h0]
        elif (self.unitary_impl=='ASB2016'):
            self.trainable_weights = [self.W, self.theta, self.reflection, self.b, self.h0]
        
        self.regularizers = []
        #if self.W_regularizer:
        #    self.W_regularizer.set_param(self.W)
        #    self.regularizers.append(self.W_regularizer)
        #if self.U_regularizer:
        #    self.U_regularizer.set_param(self.U)
        #    self.regularizers.append(self.U_regularizer)
        #if self.b_regularizer:
        #    self.b_regularizer.set_param(self.b)
        #    self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')
        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim))]

    def preprocess_input(self, x):
        if self.consume_less == 'cpu':
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2]
            timesteps = input_shape[1]
            return time_distributed_dense(x, self.W, None, self.dropout_W,
                                          input_dim, self.output_dim,
                                          timesteps)
        else:
            return x

    # override Recurrent's get_initial_states function to load the trainable
    # initial hidden state
    def get_initial_states(self, x):
            initial_state = K.expand_dims(self.h0,dim=0) # (1, output_dim)
            initial_state = K.tile(initial_state, [x.shape[0], 1])  # (samples, output_dim)
            #initial_states = [initial_state for _ in range(len(self.states))]
            initial_states = [initial_state]
            return initial_states

    def step(self, x, states):
        prev_output = states[0]
        B_U = states[1]
        B_W = states[2]

        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W)

        if (self.activation=='soft_thresh'):
            preactivation = h + K.dot(prev_output * B_U, self.Uaug)
            preactivation_abs = K.sqrt(self.epsilon + preactivation**2 + preactivation[:,self.swap_re_im]**2)
            rescale = K.maximum(preactivation_abs+self.baug,0.)/(preactivation_abs + self.epsilon)
            output = preactivation*rescale
        else:
            print "Activation",self.activation,"not implemented"
            raise NotImplementedError
        return output, [output]

    def get_constants(self, x):
        constants = []
        if 0 < self.dropout_U < 1:
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, self.output_dim))
            B_U = K.in_train_phase(K.dropout(ones, self.dropout_U), ones)
            constants.append(B_U)
        else:
            constants.append(K.cast_to_floatx(1.))
        if self.consume_less == 'cpu' and 0 < self.dropout_W < 1:
            input_shape = self.input_spec[0].shape
            input_dim = input_shape[-1]
            ones = K.ones_like(K.reshape(x[:, 0, 0], (-1, 1)))
            ones = K.tile(ones, (1, input_dim))
            B_W = K.in_train_phase(K.dropout(ones, self.dropout_W), ones)
            constants.append(B_W)
        else:
            constants.append(K.cast_to_floatx(1.))
        return constants

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'inner_init': self.inner_init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'U_regularizer': self.U_regularizer.get_config() if self.U_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U}
        base_config = super(uRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class complex_RNN_wrapper(Layer):
    '''Unitary RNN where the output is to be fed back to input, the
       hidden state is complex-valued, and the recurrence matrix
       is unitary. Input transform is complex-valued.

       Wraps the Theano implementation of uRNN by 
       [Arjovsky,Shah,Bengio 2016], available from
       https://github.com/amarshah/complex_RNN,
       and further modified by Scott Wisdom (swisdom@uw.edu).
       
       unitary_impl: implementation of unitary recurrence matrix
            Options: 
            'ASB2016'     : uses parameterization of [Arjovsky,Shah,Bengio 2016]
            'ASB2016_fast': faster version of 'ASB2016'
            'full'        : uses full unitary matrix without unitary constraint
                            during optimization
            'full_natGrad': uses full unitary matrix with natural gradient step
                           (requires using <optimizer>_and_natGrad optimizer)
            'full_natGradRMS': uses full unitary matrix with natural gradient step
                               and RMSprop-stype regularization of gradients
    '''
    def __init__(self, output_dim, hidden_dim=None, unitary_impl='adhoc', **kwargs):
        self.output_dim = output_dim
        if hidden_dim is None:
            hidden_dim = output_dim
        self.hidden_dim=hidden_dim
        self.unitary_impl=unitary_impl
        super(complex_RNN_wrapper, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[-1]

    def call(self, x, mask=None):
        input_dim = self.input_dim
        input_type='real'
        out_every_t=False
        loss_function='MSE'
        output_type='real'
        flag_feed_forward=False
        flag_use_mask=False
        hidden_bias_mean=np.float32(0.0)
        hidden_bias_init='zero'
        Wimpl=self.unitary_impl
        if ('full' in Wimpl):
            Wimpl='full'
        elif (Wimpl=='ASB2016'):
            Wimpl='adhoc'
            #hidden_bias_init='rand'
        elif (Wimpl=='ASB2016_fast'):
            Wimpl='adhoc_fast'
        n_layers=1
        seed=1234
        x_spec=K.permute_dimensions(x,(1,0,2))
        inputs, parameters, costs = models.complex_RNN(input_dim, self.hidden_dim, self.output_dim, input_type=input_type,out_every_t=out_every_t, loss_function=loss_function,output_type=output_type,flag_feed_forward=flag_feed_forward,flag_return_lin_output=True,x_spec=x_spec,flag_use_mask=flag_use_mask,hidden_bias_mean=hidden_bias_mean,Wimpl=Wimpl,flag_return_hidden_states=True,n_layers=n_layers,seed=seed,hidden_bias_init=hidden_bias_init)

        lin_output=costs[2]
        #self.hidden_states=costs[3]

        if (self.unitary_impl=='full'):
            # just use lrng for learning rate on this parameter
            parameters[-1].name+='full_natGrad'
        elif (self.unitary_impl=='full_natGrad'):
            # use fixed lrng with natural gradient update
            parameters[-1].name+='_natGrad_unitaryAug'
        elif (self.unitary_impl=='full_natGradRMS'):
            # use fixed lrng with natural gradient update and RMSprop-style gradient adjustment
            parameters[-1].name+='_natGradRMS_unitaryAug'
        elif (self.unitary_impl=='full_enforceComplex'):
            # swap out 2Nx2N augmented unitary matrix for Nx2N, which ensures the 
            # complex number constraint is satisfied 
            parameters[-1].name+='full_natGrad'
            Waug=parameters[-1]
            WReIm=K.variable(value=Waug[:Waug.shape[1]/2,:].eval(),name=Waug.name)
            WaugFull=K.concatenate( (WReIm, K.concatenate((-WReIm[:,WReIm.shape[1]/2:],WReIm[:,:WReIm.shape[1]/2]),axis=1)),axis=0 )
            lin_output_new = theano.clone(lin_output,replace={parameters[-1]:WaugFull})
            lin_output = lin_output_new
            parameters[-1]=WReIm

        self.trainable_weights = parameters
            
        return lin_output

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

class DenseUnitaryAug(Layer):
    '''A dense unitary ReIm augmented layer
    ```
    # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        activation: name of activation function to use
            (see [activations](../activations.md)),
            or alternatively, elementwise Theano function.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: a(x) = x).
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        W_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the main weights matrix.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        W_constraint: instance of the [constraints](../constraints.md) module
            (eg. maxnorm, nonneg), applied to the main weights matrix.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer).
            This argument (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    '''
    def __init__(self, output_dim, init='svd', activation='linear', weights=None,
                 input_type='complex',
                 W_regularizer=None, b_regularizer=None, activity_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, input_dim=None, **kwargs):
        if (init=='svd'):
            self.init=unitary_svd_init
        elif (init=='ASB2016'):
            self.init=unitary_ASB2016_init
        else:
            print "Unitary recurrence initialization '%s' not supported" % inner_init
            raise NotImplementedError
        activation='linear'
        self.activation = activations.get(activation)
        self.output_dim = 2*output_dim
        if input_dim is None:
            input_dim=output_dim
        if (input_type=='real'):
            self.input_dim = input_dim
        elif (input_type=='complex'):
            self.input_dim = 2*input_dim
        else:
            print "Input type of '%s' not supported" % input_type
            raise NotImplementedError
        self.input_type = input_type

        """
        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)
        """
        self.W_regularizer = None
        self.b_regularizer = None
        self.activity_regularizer = None

        self.W_constraint = None
        self.b_constraint = None
        
        #self.bias = bias
        self.bias = False
        self.initial_weights = weights
        self.input_spec = [InputSpec(ndim=2)]

        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(DenseUnitaryAug, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = [InputSpec(dtype=K.floatx(),
                                     shape=(None, input_dim))]

        W = self.init((self.output_dim/2, self.output_dim/2))
        W = W.get_value()
        Waug = augRight(W,module=np)
        self.Waug=K.variable(Waug,name='{}_Waug_full_natGrad_unitaryAug'.format(self.name))
        self.WaugUse=self.Waug
        if (self.input_type=='real'):
            self.WaugUse = self.Waug[:self.output_dim/2,:]
        """
        if self.bias:
            self.b = K.zeros((self.output_dim,),
                             name='{}_b'.format(self.name))
            self.trainable_weights = [self.W, self.b]
        
        else:
        """
        self.trainable_weights = [self.Waug]

        self.regularizers = []
        """
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)

        if self.bias and self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.activity_regularizer:
            self.activity_regularizer.set_layer(self)
            self.regularizers.append(self.activity_regularizer)
        """

        self.constraints = {}
        """
        if self.W_constraint:
            self.constraints[self.W] = self.W_constraint
        if self.bias and self.b_constraint:
            self.constraints[self.b] = self.b_constraint
        """

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        output = K.dot(x, self.WaugUse)
        if self.bias:
            output += self.b
        return self.activation(output)

    def get_output_shape_for(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': self.init.__name__,
                  'activation': self.activation.__name__,
                  'W_regularizer': self.W_regularizer.get_config() if self.W_regularizer else None,
                  'b_regularizer': self.b_regularizer.get_config() if self.b_regularizer else None,
                  'activity_regularizer': self.activity_regularizer.get_config() if self.activity_regularizer else None,
                  'W_constraint': self.W_constraint.get_config() if self.W_constraint else None,
                  'b_constraint': self.b_constraint.get_config() if self.b_constraint else None,
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(DenseUnitaryAug, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class tanhAug(Layer):
    '''tanh on magnitude of ReIm augmented complex vector, copy phase through
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        theta: float >= 0. Threshold location of activation.
    # References
    '''
    def __init__(self, flag_clip=False, **kwargs):
        self.epsilon=1e-5
        self.flag_clip = flag_clip
        if self.flag_clip:
            self.clip_min=0.0
            self.clip_max=T.arctanh(np.float32(1-3e-8)).eval()
        super(tanhAug, self).__init__(**kwargs)

    def build(self, input_shape):
        self.swap_re_im = build_swap_re_im(input_shape[1]/2)

    def call(self, x, mask=None):
        x_abs = K.sqrt(self.epsilon + x**2 + x[:,self.swap_re_im]**2)
        if self.flag_clip:
            x_abs = K.clip(x_abs,self.clip_min,self.clip_max)
        rescale = K.tanh(x_abs)/(x_abs + self.epsilon)
        return rescale * x
    def get_output_shape_for(self, input_shape):
        return input_shape
    """
    def get_config(self):
        config = {}
        base_config = super(tanhAug, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    """

class arctanhAug(Layer):
    '''arctanh on magnitude of ReIm augmented complex vector, copy phase through
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as the input.
    # Arguments
        theta: float >= 0. Threshold location of activation.
    # References
    '''
    def __init__(self, **kwargs):
        self.epsilon=1e-5
        super(arctanhAug, self).__init__(**kwargs)

    def build(self, input_shape):
        self.swap_re_im = build_swap_re_im(input_shape[1]/2)

    def call(self, x, mask=None):
        x_abs = K.sqrt(self.epsilon + x**2 + x[:,self.swap_re_im]**2)
        x_abs = K.clip(x_abs,0.,1-3e-8)
        rescale = T.arctanh(x_abs)/(x_abs + self.epsilon)
        return rescale * x

    def get_output_shape_for(self, input_shape):
        return input_shape
    """
    def get_config(self):
        config = {}
        base_config = super(arctanhAug, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    """
