import theano
import theano.tensor as T
import numpy as np


def clipped_gradients(gradients, gradient_clipping):
    clipped_grads = [T.clip(g, -gradient_clipping, gradient_clipping)
                     for g in gradients]
    return clipped_grads

def gradient_descent(learning_rate, parameters, gradients):        
    updates = [(p, p - learning_rate * g) for p, g in zip(parameters, gradients)]
    return updates

def gradient_descent_momentum(learning_rate, momentum, parameters, gradients):
    velocities = [theano.shared(np.zeros_like(p.get_value(), 
                                              dtype=theano.config.floatX)) for p in parameters]

    updates1 = [(vel, momentum * vel - learning_rate * g) 
                for vel, g in zip(velocities, gradients)]
    updates2 = [(p, p + vel) for p, vel in zip(parameters, velocities)]
    updates = updates1 + updates2
    return updates 


def rms_prop(learning_rate, parameters, gradients, idx_project=None):        
    rmsprop = [theano.shared(1e-3*np.ones_like(p.get_value())) for p in parameters]

    if idx_project is not None:
        # we will use projected gradient on the Stiefel manifold on these parameters
        # we will assume these parameters are unitary matrices in real-composite form
        parameters_proj = [parameters[i] for i in idx_project]
        gradients_proj  = [gradients[i] for i in idx_project]
        sizes_proj = [p.shape for p in parameters_proj]
        # compute gradient in tangent space of Stiefel manifold (see Lemma 4 of [Tagare 2011])
        # X = A+jB
        Aall = [T.cast(T.transpose(p[:s[0]/2,:s[0]/2]),'complex64') for s, p in zip(sizes_proj, parameters_proj)]
        Ball = [T.cast(T.transpose(p[:s[0]/2,s[0]/2:]),'complex64') for s, p in zip(sizes_proj, parameters_proj)]
        # G = C+jD
        Call = [T.cast(T.transpose(g[:s[0]/2,:s[0]/2]),'complex64') for s, g in zip(sizes_proj, gradients_proj)]
        Dall = [T.cast(T.transpose(g[:s[0]/2,s[0]/2:]),'complex64') for s, g in zip(sizes_proj, gradients_proj)]
        # GX^H = CA^T + DB^T + jDA^T -jCB^T
        GXHall = [T.dot(C,T.transpose(A)) + T.dot(D,T.transpose(B))  \
               + T.cast(1j,'complex64')*T.dot(D,T.transpose(A)) - T.cast(1j,'complex64')*T.dot(C,T.transpose(B)) \
               for A, B, C, D in zip(Aall, Ball, Call, Dall)]
        Xall = [A+T.cast(1j,'complex64')*B for A, B in zip(Aall, Ball)]
        ## Gt = (GX^H - XG^H)X
        #Gtall = [T.dot(GXH - T.transpose(T.conj(GXH)),X) for GXH, X in zip(GXHall,Xall)]
        # compute Cayley transform, which is curve of steepest descent (see section 4 of [Tagare 2011])
        Wall = [GXH - T.transpose(T.conj(GXH)) for GXH in GXHall]
        Iall = [T.identity_like(W) for W in Wall]
        W2pall = [I+(learning_rate/T.cast(2,'complex64'))*W for I, W in zip(Iall,Wall)]
        W2mall = [I-(learning_rate/T.cast(2,'complex64'))*W for I, W in zip(Iall,Wall)]
        if (learning_rate>0.0):
            Gtall = [T.dot(T.dot(T.nlinalg.matrix_inverse(W2p),W2m),X) for W2p, W2m, X in zip(W2pall, W2mall, Xall)]
        else:
            Gtall = [X for X in Xall]
        # perform transposes to prepare for converting back to transposed real-composite form
        GtallRe = [T.transpose(T.real(Gt)) for Gt in Gtall]
        GtallIm = [T.transpose(T.imag(Gt)) for Gt in Gtall]
        # convert back to real-composite form:
        gradients_tang = [T.concatenate( [T.concatenate([GtRe,GtIm], axis=1), T.concatenate([(-1)*GtIm,GtRe], axis=1)], axis=0) for GtRe, GtIm in zip(GtallRe,GtallIm)]

    new_rmsprop = [0.9 * vel + 0.1 * (g**2) for vel, g in zip(rmsprop, gradients)]

    updates1 = zip(rmsprop, new_rmsprop)
    updates2 = [(p, p - learning_rate * g / T.sqrt(rms)) for 
                p, g, rms in zip(parameters, gradients, new_rmsprop)]
    if idx_project is not None:
        # project back on to the Stiefel manifold using SVD
        # see 3.3 of [Absil and Malick 2012]
        def proj_stiefel(X):
            # projects a square transposed real-composite form matrix X onto the Stiefel manifold
            n=X.shape[0]
            # X=A+jB
            A=T.transpose(X[:n/2,:n/2])
            B=T.transpose(X[:n/2,n/2:])
            U,S,V = T.nlinalg.svd(A+T.cast(1j,'complex64')*B)
            W=T.dot(U,V)
            # convert back to transposed real-composite form
            WRe = T.transpose(T.real(W))
            WIm = T.transpose(T.imag(W))
            Wrc = T.concatenate( [T.concatenate([WRe,WIm], axis=0), T.concatenate([(-1)*WIm,WRe], axis=0)], axis=1)
            return Wrc
        
        new_rmsprop_proj = [new_rmsprop[i] for i in idx_project] 
        #updates2_proj=[(p,proj_stiefel(p - learning_rate * g )) for
        #               p, g, rms in zip(parameters_proj,gradients_tang, new_rmsprop_proj)]
        updates2_proj=[(p, g ) for
                       p, g, rms in zip(parameters_proj,gradients_tang, new_rmsprop_proj)]
        for i in range(len(updates2_proj)):
            updates2[idx_project[i]]=updates2_proj[i]
    
    updates = updates1 + updates2

    return updates, rmsprop
 
