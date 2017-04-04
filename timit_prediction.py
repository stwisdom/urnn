import sys
sys.setrecursionlimit(10000)
import cPickle
import gzip
import theano
import pdb
from fftconv import cufft, cuifft
import numpy as np
import theano.tensor as T
from theano.ifelse import ifelse
from models import *
from optimizations import *
import argparse, timeit, time
import os
import scipy
import scipy.io.wavfile
import scipy.fftpack as fft
import scipy.signal
import scipy.linalg
import librosa
from util import (stft_mc,iAugSTFT,wavwrite)


def wavread(wavfile):
    fs,x=scipy.io.wavfile.read(wavfile) #x will be nsampl x nch
    x=np.transpose(x).astype(np.float32) #convert x to float32, transpose to nch x nsampl
    x=x/32768.0
    return x


def iAugFFT(Xaug,axis=0):
    F=Xaug.shape[axis]/2
    X=np.take(Xaug,np.arange(0,F),axis=axis)+np.complex64(1j)*np.take(Xaug,np.arange(F,2*F),axis=axis)
    X=np.concatenate((X.conj(), np.take(X,np.arange(F-2,0,-1),axis=axis)), axis=axis)
    xr=fft.ifft(X,axis=axis).real
    return xr

def load_wavfiles_names(path):

    wavfiles=list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('.wav')):
                wavfile=os.path.join(root,file)
                wavfiles.append(wavfile)

    return wavfiles


def normalize_data(x,data_normalization,data_type,mask=None,n=None):
    
    data_normalization=data_normalization.lower()
    
    if mask is None:
        mask=np.ones_like(x)

    if n is None:
        n=x.shape[2]

    stats={}

    if ('perutt' in data_normalization):
        axes_mean=()
    else:
        axes_mean=(1)

    if ('mean' in data_normalization):
        # means of input data z
        stats['mean']=np.mean( np.sum(x*mask[:,:,0:1],axis=0,keepdims=True)/np.float32(np.sum(mask[:,:,0:1],axis=0,keepdims=True)), axis=axes_mean)
        x=x-(stats['mean']*mask[:,:,0:1])

    if ('var' in data_normalization):
        # std devs of input data z
        if (data_type=='real'):
            x_var=np.sum( (x**2)*mask[:,:,0:1],axis=0,keepdims=True)/np.float32(np.sum(mask[:,:,0:1],axis=0,keepdims=True))
        elif (data_type=='complex'):
            x_var=np.sum( (x[:,:,:n]**2+x[:,:,n:]**2)*mask[:,:,0:1],axis=0,keepdims=True)/np.float32(np.sum(mask[:,:,0:1],axis=0,keepdims=True))
        
        stats['std']=np.sqrt(np.mean(x_var,axis=axes_mean))
        if (data_type=='real'):
            x=x*mask[:,:,0:1]/(np.float32(1e-7)+stats['std'])
        elif (data_type=='complex'):
            x=x*mask[:,:,0:1]/(np.float32(1e-7)+np.sqrt(2).astype(np.float32)*np.tile(stats['std'],(1,1,2)))

    return x, stats


def generate_data(wavfiles,params_stft,prng,flag_unwrap_phase=True):
    N=params_stft['N']
    hop=params_stft['hop']
    nch=params_stft['nch']
    window=params_stft['window']
    F=N/2+1

    # initialize matrices to hold concatenated STFTs
    X=np.zeros((nch*F,0)).astype(np.complex64)
    Y=np.zeros((nch*F,0)).astype(np.complex64)

    # initialize frame indices for individual files
    fidx=np.zeros((len(wavfiles),2)).astype(np.int32)
    ifidx=0
    ifile=0
    for wavfile in wavfiles:
        print "Read file %d of %d total: %s" % (ifile+1,len(wavfiles),wavfile)
        # read in reference output audio
        y=wavread(wavfile)
        Ycur=stft_mc(y,N,hop,window)
        Ycur=Ycur[:,:,:nch] #restrict to desired number of channels
        Ycur=np.transpose(Ycur,(0,2,1)) #is now F x nch x nfram
        Ycur=np.reshape(Ycur,(nch*F,Ycur.shape[2]),order='F') #stack multiple channels in first dimension
        # update frame indices for this file
        nfram=Ycur.shape[1]
        fidx[ifile,0]=ifidx
        ifidx+=nfram
        fidx[ifile,1]=ifidx
        ifile+=1
        if flag_unwrap_phase:
            # remove window hop phases:
            Yphase=np.float32(np.unwrap(np.angle(Ycur),axis=1))
            frange=np.arange(0,F,dtype=np.float32)/N
            trange=np.arange(0,nfram,dtype=np.float32)*hop
            Yphase=Yphase-2*np.pi*np.outer(frange,trange)
            Ycur=np.abs(Ycur)*np.exp(1j*Yphase)
        # add Y to total data
        Y=np.concatenate((Y,Ycur),axis=1)

    Xaug=prng.randn(2*F,Y.shape[1])/np.sqrt(2) #unit variance circular complex Gaussians
    Yaug=np.concatenate((np.real(Y),np.imag(Y)),axis=0)
    return Xaug,Yaug,fidx


def generate_synth_data(n_seq,time_steps,sizes,prng,Winit='svd'):
    n_input=sizes['n_input']
    n_hidden=sizes['n_hidden']
    Xaug=prng.randn(2*n_input,n_seq*time_steps).astype(np.float32)/np.sqrt(2) #unit variance circular complex Gaussians in real-composite form
    if (Winit=='svd'):
        W=prng.randn(n_hidden,n_hidden).astype(np.complex64)+1j*prng.randn(n_hidden,n_hidden).astype(np.complex64)
        U, S, V = np.linalg.svd(W)
        W = np.dot(U,V)
        # convert W to real-composite form for right multiplication
        #  real-composite for right multiplication, g=h^T W, with h=x+jy and W=A+jB,
        #  is grc=hrc^T Wrc, with Wrc=[A^T, B^T; -B^T, A^T] and hrc=[x; y]
        #
        #  real-composite for left multiplication, g=Wh, with h=x+jy and W=A+jB,
        #  is grc=Wrc hrc, with Wrc=[A, -B; B, A] and hrc=[x; y]
        A=np.transpose(np.real(W))
        B=np.transpose(np.imag(W))
        Wr   = np.concatenate( [     A, B], axis=1) #create [ A, B]
        Wc   = np.concatenate( [(-1)*B, A], axis=1) #create [-B, A]
        Waug = np.concatenate( [Wr,Wc], axis=0) # create [A,B; -B, A]
    elif (Winit=='adhoc'):
        Wparams=initialize_unitary(n_hidden,'full',prng)
        Waug = Wparams[0].get_value()
    elif (Winit=='adhoc2x'):
        Wparams1=initialize_unitary(n_hidden,'full',prng)
        Waug1 = Wparams1[0]
        Waug1np = Waug1.get_value()
        Waug1np = Waug1np[:n_hidden,:] # only take first row of blocks to get correct augmented form after multiplication within numerical precision
        Wparams2=initialize_unitary(n_hidden,'full',prng)
        Waug2 = Wparams2[0]
        Waug_row1=np.dot(Waug1np,Waug2.get_value())
        Waug=np.concatenate([ Waug_row1, np.concatenate([-Waug_row1[:,n_hidden:],Waug_row1[:,:n_hidden]],axis=1) ],axis=0)

    fidx0 = np.arange(0,n_seq*time_steps,time_steps)
    fidx1 = np.arange(time_steps,n_seq*time_steps+time_steps,time_steps)
    fidx  = np.concatenate( [np.reshape(fidx0,(n_seq,1)), np.reshape(fidx1,(n_seq,1))] , axis=1)
    return Xaug, Waug, fidx


def main(n_iter, n_batch, n_hidden, learning_rate, savefile, model, input_type, out_every_t, loss_function, fold, scene, n_reflections=None,flag_telescope=True,nch=1,flag_unwrap_phase=True,indir="audio_8khz",outdir=None,dataset="timit",initfile=None,flag_feed_forward=True,flag_generator=False,downsample_train=1,downsample_test=1,time_steps=None,n_Givens=None,prng_seed_Givens=52016,num_allowed_test_inc=10,iters_per_validCheck=20,flag_useFullW=False,flag_onlyOptimW=True,lam=np.float32(0.0),Vnorm=np.float32(0.0),Unorm=np.float32(0.0),n_layers=1,num_pred_steps=0,hidden_bias_mean=0.1,data_transform='',bwe_frac=np.float32(1.0),data_normalization='none',offset_eval=None,olap=50,window=None,flag_noDiv=0,flag_noComplexConstraint=0,Winit='svd',seed=1234,optim_alg="rmsprop",n_utt_eval_spec=-1):

    if offset_eval<0:
        offset_eval=None

    cost_weight=None
    cost_transform=None
    
    # --- Set data params ----------------
    if (dataset=='timit16'):
        N=512 #32 ms at fs=16kHz
    else:
        N=256 #32 ms at fs=8kHz
    hop=np.round(np.float32(N)*np.float32(100.0-olap)/100.0).astype(np.int)
    if (window=='hann'):
        window=scipy.signal.hann(N,sym=False)
    elif (window=='sqrt_hann'):
        window=np.sqrt(scipy.signal.hann(N,sym=False))
    else:
        window=None
    params_stft={'N': N, 'hop': hop, 'nch': nch, 'window': window}  #STFT parameters
    F=N/2+1
    n_input =F         #we're stacking multiple channels on top of each other
    n_output=n_input   #because we are building an autoencoder

    ds_train = 1 #downsampling factor for training
    ds_test = 1 #downsampling factor for test

    #set paths:
    if (dataset=='timit') or (dataset=='timit_trainNoSA_dev_coreTest'):
        path_train=''.join(["/data1/timit/TIMIT_8khz/TRAIN"])
        path_test =''.join(["/data1/timit/TIMIT_8khz/TEST"])
    elif (dataset=='timit16'):
        path_train=''.join(["/data1/timit/TIMIT_16khz/TRAIN"])
        path_test =''.join(["/data1/timit/TIMIT_16khz/TEST"])
    elif (dataset=='synthgen'):
        path_train=None
        path_test=None
    else:
        raise ValueError("dataset must be synthgen or timit")

    if not (dataset == 'synthgen'):
        #load and downsample wavfiles list for training
        wavfiles_train_all=load_wavfiles_names(path_train)
        wavfiles_train    =wavfiles_train_all[::ds_train]
        n_train=len(wavfiles_train)

        #load and downsample wavfiles list for test
        wavfiles_test_all =load_wavfiles_names(path_test)
        wavfiles_test     =wavfiles_test_all[::ds_test]
    else:
        n_train=int(2e4)
        n_test =int(2e3)
        n_input=n_hidden
        n_output=n_hidden
        sizes  ={'n_input': n_input, 'n_hidden': n_hidden, 'n_output': n_output}

    num_batches = int(n_train / n_batch)


    # --- Create data --------------------

    # set up random number generators for repeatable results
    train_prng = np.random.RandomState(5678)
    test_prng = np.random.RandomState(42)

    # generate and/or load the data
    savefile_timit_data=None
    if (dataset=='timit'):
        savefile_timit_data='timit_data'
    elif (dataset=='timit16'):
        savefile_timit_data='/data1/swisdom/timit16_data'
    elif (dataset=='timit_trainNoSA_dev_coreTest'):
        savefile_timit_data='timit_data_trainNoSA_dev_coreTest'

    if ('timit' in dataset) and (os.path.isfile(savefile_timit_data) or os.path.isfile(savefile_timit_data+'_train_xdata_stack')):
        # we're using TIMIT and a save file for TIMIT exists, so load up the data
        print "Save file %s for TIMIT data exists, loading it from the hard drive..." % savefile_timit_data
        if (dataset=='timit') or (dataset=='timit_trainNoSA_dev_coreTest'):
            L=cPickle.load(file(savefile_timit_data,'r'))
            print "Loaded TIMIT data"
            train_z_stack=L['train_z_stack']
            train_xdata_stack=L['train_xdata_stack']
            fidx_train=L['fidx_train']
            test_z_stack=L['test_z_stack']
            test_xdata_stack=L['test_xdata_stack']
            fidx_test=L['fidx_test']
        elif (dataset=='timit16'):
            for key in ['train_z_stack','train_xdata_stack','fidx_train','test_z_stack','test_xdata_stack','fidx_test']:
                print "Broken exec statement"
                #exec("%s=np.load(file(savefile_timit_data+'_'+key,'rb'))" % key)
        n_train=fidx_train.shape[0]
        num_batches = int(n_train / n_batch)
    elif not (dataset == 'synthgen'):
        # we aren't using the synthgen dataset, or we aren't using the timit dataset, 
        # or the savefile for timit data doesn't exist, so load data using the lists
        # of wavfiles and generate associated random data
        
        if (dataset=='timit_trainNoSA_dev_coreTest'):
            # adjust wavfiles lists to exclude SA utterances from train
            # and make the test set concatenated TIMIT dev set and
            # core test set
            wavfiles_train = [x for x in wavfiles_train if (not ('sa' in x.lower()))]
            wavfiles_test = [x for x in wavfiles_test if (not ('sa' in x.lower()))]
            speakers_dev = [line.rstrip('\n') for line in open('timit_dev_spk.list')]
            wavfiles_dev = [x for x in wavfiles_test if any(speaker in x.lower() for speaker in speakers_dev)]
            speakers_coreTest = [line.rstrip('\n') for line in open('timit_test_spk.list')]
            wavfiles_coreTest = [x for x in wavfiles_test if any(speaker in x.lower() for speaker in speakers_coreTest)]
            wavfiles_extraTest = [x for x in wavfiles_test if (not (x in wavfiles_dev+wavfiles_coreTest))]
            wavfiles_test=wavfiles_dev+wavfiles_coreTest+wavfiles_extraTest
        train_z_stack, train_xdata_stack, fidx_train = generate_data(wavfiles_train,
                                                                     params_stft,
                                                                     train_prng,
                                                                     flag_unwrap_phase)
        test_z_stack,  test_xdata_stack,  fidx_test  = generate_data(wavfiles_test,
                                                                     params_stft,
                                                                     test_prng,
                                                                     flag_unwrap_phase)
        # z are 2*nch*F x \sum_utt n_fram(utt), xdata are 2*nsrc*nch*F x \sum_utt n_fram(utt)

        # if we're doing TIMIT and the save file doesn't exist, write it out:
        if ( 'timit' in dataset ) and not (os.path.isfile(savefile_timit_data) or os.path.isfile(savefile_timit_data+'_train_xdata_stack')):
            print "Saving TIMIT data to file %s" % savefile_timit_data
            # we have read in and generated z's for TIMIT data; save it off
            save_vals_timit_data={'train_z_stack': train_z_stack, 
                                   'train_xdata_stack': train_xdata_stack, 
                                   'fidx_train': fidx_train, 
                                   'test_z_stack': test_z_stack, 
                                   'test_xdata_stack': test_xdata_stack, 
                                   'fidx_test': fidx_test}
            #if (dataset=='timit'):
            if not (dataset=='timit16'):
                cPickle.dump(save_vals_timit_data, file(savefile_timit_data, 'wb'), cPickle.HIGHEST_PROTOCOL)
            #elif (dataset=='timit16'):
            else:
                for key in save_vals_timit_data.keys():
                    np.save(file(savefile_timit_data+'_'+key,'wb'),save_vals_timit_data[key])
            """
            else:
                print "Unknown timit dataset name %s" % dataset 
                return
            """
    else:
        # we're using the synthgen dataset, so use a different function, generate_synth_data, to create data
        train_z_stack, synth_Waug, fidx_train = generate_synth_data(n_train,time_steps,sizes,train_prng,Winit)
        #train_xdata_stack = np.zeros_like(train_z_stack) #set xdata to 0, since we'll generate these later
        train_xdata_stack = np.zeros((2*n_output,n_train*time_steps))
        test_z_stack,  extra_Waug, fidx_test  = generate_synth_data(n_test, time_steps,sizes,test_prng,Winit)
        #test_xdata_stack = np.zeros_like(test_z_stack)
        test_xdata_stack = np.zeros((2*n_output,n_test*time_steps))

    # if input z is real-valued,
    #     train_z_stack should be of dimension n_framMax x n_input    x n_utt
    # if input z is complex-valued,
    #     train_z_stack should be of dimension n_framMax x 2*n_input  x n_utt
    # if xdata is real-valued,
    #     train_xdata_stack should be of dimension n_framMax x n_output   x n_utt
    # if xdata is complex-valued,
    #     train_xdata_stack should be of dimension n_framMax x 2*n_output x n_utt


    # check if we're doing an autoencoder (flag_generator=False) or a generator (flag_generator=True)
    if not flag_generator:
        # since we're doing an autoencoder, set train_z equal to train_xdata 
        train_z_stack=np.copy(train_xdata_stack)
        test_z_stack=np.copy(test_xdata_stack)


    #tweaks to ensure dynamical system output is relatively stable (determined by playing with parameters in Matlab):
    ## scale random data down for timit
    #if (dataset == 'timit'):
    #    train_z_stack = train_z_stack*np.sqrt(1.0)
    #    test_z_stack = test_z_stack*np.sqrt(1.0)
    # use a negative hidden bias mean
    if (dataset == 'synthgen'):
        hidden_bias_mean=-0.1
    #else:
    #    hidden_bias_mean=0.0

    # create padded inputs and outputs for train:
    lens_train=fidx_train[:,1]-fidx_train[:,0]
    n_framMax_train=np.max(lens_train)
    n_utt_train=len(lens_train)
    train_z = np.zeros((n_framMax_train,2*n_input, n_utt_train)).astype(np.float32)
    train_xdata = np.zeros((n_framMax_train,2*n_output,n_utt_train)).astype(np.float32)
    for iutt in range(n_utt_train):
        train_z[:lens_train[iutt],:,iutt]=np.transpose(train_z_stack[:,fidx_train[iutt,0]:fidx_train[iutt,1]])
        train_xdata[:lens_train[iutt],:,iutt]=np.transpose(train_xdata_stack[:,fidx_train[iutt,0]:fidx_train[iutt,1]])
    # train_z is in augmented form and is now of dimensions n_framMax_train x 2*n_input  x n_utt_train
    # train_xdata is in augmented form and is now of dimensions n_framMax_train x 2*n_output x n_utt_train

    # create padded inputs and outputs for test:
    lens_test=fidx_test[:,1]-fidx_test[:,0]
    n_framMax_test=np.max(lens_test)
    n_utt_test=len(lens_test)
    test_z = np.zeros((n_framMax_test,2*n_input, n_utt_test)).astype(np.float32)
    test_xdata = np.zeros((n_framMax_test,2*n_output,n_utt_test)).astype(np.float32)
    for iutt in range(n_utt_test):
        test_z[:lens_test[iutt],:,iutt]=np.transpose(test_z_stack[:,fidx_test[iutt,0]:fidx_test[iutt,1]])
        test_xdata[:lens_test[iutt],:,iutt]=np.transpose(test_xdata_stack[:,fidx_test[iutt,0]:fidx_test[iutt,1]])
    # test_z is in augmented form and is now of dimensions n_framMax_test x 2*n_input  x n_utt_test
    # test_xdata is in augmented form and is now of dimensions n_framMax_test x 2*n_output x n_utt_test

    # to get scan to work properly, transpose x and y to be of size n_framMax x n_utt x n_<input,output>
    train_z=np.transpose(train_z,[0,2,1])
    train_xdata=np.transpose(train_xdata,[0,2,1])
    test_z =np.transpose(test_z,[0,2,1])
    test_xdata =np.transpose(test_xdata,[0,2,1])

    output_type='complex' #assume complex-valued data
    if (data_transform=='logmag'):
        print "Using log-magnitude transform on input and output data"
        print ""
        input_type='real'
        output_type='real'
        train_z=10.0*np.log10(1e-5 + train_z[:,:,:n_input]**2 + train_z[:,:,n_input:]**2)
        train_xdata=10.0*np.log10(1e-5 + train_xdata[:,:,:n_output]**2 + train_xdata[:,:,n_output:]**2)
        test_z=10.0*np.log10(1e-5 + test_z[:,:,:n_input]**2 + test_z[:,:,n_input:]**2)
        test_xdata=10.0*np.log10(1e-5 + test_xdata[:,:,:n_output]**2 + test_xdata[:,:,n_output:]**2)
    elif (data_transform=='logmag_phasePrediction'):
        print "Using log-magnitude transform on input data, using linear complex for output, modifying cost function for phase prediction"
        print ""
        input_type='real'
        output_type='real'
        train_z=10.0*np.log10(1e-5 + train_z[:,:,:n_input]**2 + train_z[:,:,n_input:]**2)
        test_z=10.0*np.log10(1e-5 + test_z[:,:,:n_input]**2 + test_z[:,:,n_input:]**2)
        cost_transform='magTimesPhase'
        loss_function="none_in_scan"
    elif (data_transform=='time_domain_windowed'):
        print "Using windowed time-domain frames for input and output data"
        print ""
        input_type='real'
        output_type='real'
        n_input=N
        n_output=N
        
        start_time=time.time()
        train_z=iAugFFT(train_z,axis=2)
        train_xdata=np.copy(train_z)
        #train_xdata=iAugFFT(train_xdata,axis=2)
        test_z=iAugFFT(test_z,axis=2)
        test_xdata=np.copy(test_z)
        #test_xdata=iAugFFT(test_xdata,axis=2)
        elapsed_time = time.time() - start_time
        print "Elapsed time to compute IFFTs: %f" % elapsed_time
        print ""
        

    if (bwe_frac<1.0):
        if not (n_input==n_output):
            print "Error: bwe_frac is less than 1.0, but n_input and n_output are not equal! Exiting..."
            return
        bwe_n_output = np.round(bwe_frac*n_output)
        bwe_n_input = n_output - bwe_n_output
        # grab lower indices of z as input
        train_z=np.concatenate( [train_z[:,:,:bwe_n_input],train_z[:,:,n_input:n_input+bwe_n_input]],axis=2)
        test_z=np.concatenate( [test_z[:,:,:bwe_n_input],test_z[:,:,n_input:n_input+bwe_n_input]],axis=2)
        # grab upper indices of xdata as targets
        train_xdata=np.concatenate( [train_xdata[:,:,bwe_n_input:n_output],train_xdata[:,:,n_output+bwe_n_input:]], axis=2)
        test_xdata=np.concatenate( [test_xdata[:,:,bwe_n_input:n_output],test_xdata[:,:,n_output+bwe_n_input:]], axis=2)
        n_input=bwe_n_input
        n_output=bwe_n_output

    if (num_pred_steps>0):
        print "Predicting reference data %d steps ahead" % num_pred_steps
        print ""
        train_xdata[:n_framMax_train-num_pred_steps,:,:]=train_xdata[num_pred_steps:,:,:]
        test_xdata[:n_framMax_test-num_pred_steps,:,:]=test_xdata[num_pred_steps:,:,:]

    # apply downsampling factors to train and test data, if the factors are greater than 1
    if (downsample_train>1):
        train_z=train_z[:,0:n_utt_train:downsample_train,:]
        train_xdata=train_xdata[:,0:n_utt_train:downsample_train,:]
        lens_train=lens_train[0:n_utt_train:downsample_train]
        num_batches=num_batches/downsample_train
        n_train=n_train/downsample_train
        n_utt_train=n_utt_train/downsample_train
    
    if offset_eval is not None:
        if (downsample_test==1):
            # build eval data:
            if (n_utt_eval_spec>0):
                n_utt_eval=n_utt_eval_spec
                eval_z=test_z[:,offset_eval:offset_eval+n_utt_eval,:]
                eval_xdata=test_xdata[:,offset_eval:offset_eval+n_utt_eval,:]
                lens_eval=lens_test[offset_eval:offset_eval+n_utt_eval]
            else:    
                eval_z=test_z[:,offset_eval:n_utt_test,:]
                eval_xdata=test_xdata[:,offset_eval:n_utt_test,:]
                lens_eval=lens_test[offset_eval:n_utt_test]
                n_utt_eval=n_utt_test-offset_eval
            # clip test data:
            test_z=test_z[:,:offset_eval,:]
            test_xdata=test_xdata[:,:offset_eval,:]
            lens_test=lens_test[:offset_eval]
            n_utt_test=offset_eval
        else:
            eval_z=test_z[:,offset_eval:n_utt_test:downsample_test,:]
            eval_xdata=test_xdata[:,offset_eval:n_utt_test:downsample_test,:]
            lens_eval=lens_test[0:n_utt_test:downsample_test]
            n_utt_eval=n_utt_test/downsample_test
    
    if (downsample_test>1):
        test_z=test_z[:,0:n_utt_test:downsample_test,:]
        test_xdata=test_xdata[:,0:n_utt_test:downsample_test,:]
        lens_test=lens_test[0:n_utt_test:downsample_test]
        n_utt_test=n_utt_test/downsample_test


    # set data masks, if the data sequences have unequal length
    if ('timit' in dataset):
        flag_use_mask=True
    else:
        flag_use_mask=False

    if flag_use_mask:
        train_xdata_mask=np.zeros((n_framMax_train,n_utt_train,1),dtype=np.int8)
        for ii in xrange(lens_train.shape[0]):
            train_xdata_mask[0:lens_train[ii],ii,:]=1

        if offset_eval is not None:
            eval_xdata_mask=np.zeros((n_framMax_test,n_utt_eval,1),dtype=np.int8)
            for ii in xrange(lens_eval.shape[0]):
                eval_xdata_mask[0:lens_eval[ii],ii,:]=1
        
        test_xdata_mask=np.zeros((n_framMax_test,n_utt_test,1),dtype=np.int8)
        for ii in xrange(lens_test.shape[0]):
            test_xdata_mask[0:lens_test[ii],ii,:]=1


    # apply normalization to data, if specified
    print "Applying normalization of %s to data..." % data_normalization
    print ""
    
    stats={}

    #normalize train data
    if flag_use_mask:
        train_mask=train_xdata_mask
    else:
        train_mask=None
    train_z, stats['train_z_stats']=normalize_data(train_z,data_normalization,input_type,mask=train_mask,n=n_input)
    train_xdata, stats['train_xdata_stats']=normalize_data(train_xdata,data_normalization,output_type,mask=train_mask,n=n_output)
    
    #normalize test data
    if flag_use_mask:
        test_mask=test_xdata_mask
    else:
        test_mask=None
    test_z, stats['test_z_stats']=normalize_data(test_z,data_normalization,input_type,mask=test_mask,n=n_input)
    test_xdata, stats['test_xdata_stats']=normalize_data(test_xdata,data_normalization,output_type,mask=test_mask,n=n_output)

    if offset_eval is not None:
        #normalize eval data
        if flag_use_mask:
            eval_mask=eval_xdata_mask
        else:
            eval_mask=None
        eval_z, stats['eval_z_stats']=normalize_data(eval_z,data_normalization,input_type,mask=eval_mask,n=n_input)
        eval_xdata, stats['eval_xdata_stats']=normalize_data(eval_xdata,data_normalization,output_type,mask=eval_mask,n=n_output)
    

    # --- Create theano graph and compute gradients ----------------------

    gradient_clipping = np.float32(1)

    if (model == 'LSTM'):
        n_input_LSTM=n_input
        n_output_LSTM=n_output
        if (input_type=='complex'):
            n_input_LSTM=2*n_input
        if (output_type=='complex'):
            n_output_LSTM=2*n_output
        inputs, parameters, costs = LSTM(n_input_LSTM, n_hidden, n_output_LSTM, input_type=input_type,out_every_t=out_every_t, loss_function=loss_function,flag_use_mask=flag_use_mask,flag_return_lin_output=True,flag_return_hidden_states=True,cost_weight=cost_weight,cost_transform=cost_transform,seed=seed)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'complex_RNN'):
        if flag_useFullW:
            Wimpl='full'
        else:
            if (n_Givens is None) or (n_Givens < 1):
                Wimpl='adhoc'
            else:
                Wimpl='givens'
        
        # build computational graph for train and test
        inputs, parameters, costs = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,out_every_t=out_every_t, loss_function=loss_function,output_type=output_type,flag_feed_forward=flag_feed_forward,flag_return_lin_output=True,flag_use_mask=flag_use_mask,hidden_bias_mean=hidden_bias_mean,Wimpl=Wimpl,prng_Givens=np.random.RandomState(prng_seed_Givens),lam=lam,Vnorm=Vnorm,Unorm=Unorm,flag_return_hidden_states=True,n_layers=n_layers,cost_weight=cost_weight,cost_transform=cost_transform,flag_noComplexConstraint=flag_noComplexConstraint,seed=seed)
       
        idx_project=None
        if (dataset == 'synthgen'):
            if flag_onlyOptimW:
                # don't optimize V, U, or out_bias (elements 0, 1, and 4/3 of parameters)
                if (Wimpl=='adhoc'):
                    # reflection and theta
                    parameters_optimize=[parameters[3],parameters[5]]
                elif (Wimpl=='full'):
                    parameters_optimize=[parameters[5]]
                    idx_project=[0]
            else:
                parameters_optimize=parameters
                if (Wimpl=='full'):
                    # since we're using a full W matrix, indicate its index in the
                    # parameters_optimize list to make sure we use Steifel manifold
                    # optimization on it:
                    idx_project=[5]

            gradients = T.grad(costs[0], parameters_optimize)

            # build computational graph for generating train and test data
            inputs_synth, parameters_synth, costs_synth = complex_RNN(n_input, n_hidden, n_output, input_type=input_type,out_every_t=out_every_t, loss_function='none_in_scan',output_type='complex',flag_feed_forward=flag_feed_forward,flag_return_lin_output=True,flag_use_mask=flag_use_mask,hidden_bias_mean=-0.1,Wimpl='full',lam=lam)


        elif (( 'timit' in dataset ) and flag_generator):

            if (n_layers==1):
                print "Dataset is timit and we are running a generator with 1 layer, so we'll only optimize V, b, W, and h_0, and use initialization for U and c."
                print ""
                if (Wimpl=='adhoc'):
                    # only optimize V, hidden_bias, W parameters, h_0
                    parameters_optimize=[parameters[0],parameters[2],parameters[3],parameters[5],parameters[6]]+parameters[7:]
                elif (Wimpl=='full'):
                    # only optimize V, hidden_bias, W parameters, h_0
                    parameters_optimize=[parameters[0],parameters[2],parameters[4],parameters[5]]+parameters[6:]
                    idx_project=[3]

            gradients = T.grad(costs[0], parameters_optimize)


        else:
            if (Wimpl=='full'):
                idx_project=[5]
            gradients = T.grad(costs[0], parameters)

    elif (model == 'IRNN'):
        inputs, parameters, costs = IRNN(n_input, n_hidden, n_output, input_type=input_type,
                                         out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    elif (model == 'RNN'):
        inputs, parameters, costs = tanhRNN(n_input, n_hidden, n_output, input_type=input_type,
                                            out_every_t=out_every_t, loss_function=loss_function)
        gradients = T.grad(costs[0], parameters)
        gradients = [T.clip(g, -gradient_clipping, gradient_clipping) for g in gradients]

    else:
        print "Unsuported model:", model
        return

    # allocate shared theano variables to hold train and test data
    s_train_z = theano.shared(train_z,borrow=True)
    if (dataset=='synthgen'):
        s_train_xdata = theano.shared(np.zeros((time_steps,1,1)).astype(np.float32))
    else:
        s_train_xdata = theano.shared(train_xdata,borrow=True)

    s_test_z  = theano.shared(test_z,borrow=True)
    if (dataset=='synthgen'):
        s_test_xdata = theano.shared(np.zeros((time_steps,1,1)).astype(np.float32))
    else:
        s_test_xdata  = theano.shared(test_xdata,borrow=True)

    if offset_eval is not None:
        s_eval_z = theano.shared(eval_z,borrow=True)
        s_eval_xdata = theano.shared(eval_xdata,borrow=True)

    if (dataset=='synthgen'):
        s_synth_Waug = theano.shared(synth_Waug)

    if flag_use_mask:
        s_train_xdata_mask = theano.shared(train_xdata_mask,borrow=True)
        s_test_xdata_mask = theano.shared(test_xdata_mask,borrow=True)
        if offset_eval is not None:
            s_eval_xdata_mask = theano.shared(eval_xdata_mask,borrow=True)

    # --- Compile theano functions --------------------------------------------------

    index = T.iscalar('i')

    if (dataset == 'synthgen') or ( ('timit' in dataset) and flag_generator):
        updates, rmsprop = rms_prop(learning_rate, parameters_optimize, gradients,idx_project)
    else:
        idx_project=None #assume we aren't doing projected gradient on any parameters
        if flag_useFullW:
            idx_project=[5]
        updates, rmsprop = rms_prop(learning_rate, parameters, gradients,idx_project)

    if (optim_alg=='sgd'):
        updates = gradient_descent(learning_rate, parameters, gradients)
        rmsprop = []

    if (dataset == 'synthgen'):
        # run theano functions to generate train and test data from random inputs
        Vaug = np.zeros((n_input,2*n_hidden),dtype=np.float32)
        Vaug[:n_input,:n_hidden] = np.eye(n_hidden)
        #Vaug = test_prng.randn(n_input,2*n_hidden).astype(np.float32)
        V_synth = theano.shared(Vaug)
        Uaug = np.zeros((2*n_hidden,n_output),dtype=np.float32)
        Uaug[:n_hidden,:n_output] = np.eye(n_hidden)
        U_synth = theano.shared(Uaug)
        h_0_synth = theano.shared(np.zeros((1,2*n_hidden),dtype=np.float32))
        
        givens_synth_train = {inputs_synth[0] : s_train_z,
                              inputs_synth[1] : s_train_xdata,
                              parameters_synth[0] : V_synth,
                              parameters_synth[1] : U_synth,
                              parameters_synth[4] : h_0_synth,
                              parameters_synth[5] : s_synth_Waug}
        synth_train = theano.function([], costs_synth[2], givens=givens_synth_train)

        givens_synth_test  = {inputs_synth[0] : s_test_z,
                              inputs_synth[1] : s_test_xdata,
                              parameters_synth[0] : V_synth,
                              parameters_synth[1] : U_synth,
                              parameters_synth[4] : h_0_synth,
                              parameters_synth[5] : s_synth_Waug}
        synth_test  = theano.function([], costs_synth[2], givens=givens_synth_test)

        if offset_eval is not None:
            givens_synth_eval  = {inputs_synth[0] : s_eval_z,
                                  inputs_synth[1] : s_eval_xdata,
                                  parameters_synth[0] : V_synth,
                                  parameters_synth[1] : U_synth,
                                  parameters_synth[4] : h_0_synth,
                                  parameters_synth[5] : s_synth_Waug}
            synth_eval  = theano.function([], costs_synth[2], givens=givens_synth_eval)
        # synthesize outputs for train and test
        print "Generating outputs for train set"
        print ""
        train_y_synth = synth_train()
        train_xdata = train_y_synth
        s_train_xdata=theano.shared(train_y_synth,borrow=True)
        print "Generating outputs for test set"
        print ""
        test_y_synth  = synth_test()
        test_xdata = test_y_synth
        s_test_xdata=theano.shared(test_y_synth,borrow=True)
        if offset_eval is not None:
            print "Generating outputs for eval set"
            print ""
            eval_y_synth  = synth_eval()
            eval_xdata = eval_y_synth
            s_eval_xdata=theano.shared(eval_y_synth,borrow=True)


    # set up train and test functions for training
    if flag_use_mask:
        givens = {inputs[0] : s_train_z[:, n_batch * index : n_batch * (index + 1), :],
                  inputs[1] : s_train_xdata[:, n_batch * index : n_batch * (index + 1), :],
                  inputs[2] : s_train_xdata_mask[:, n_batch * index : n_batch * (index + 1), :]}
        givens_test = {inputs[0] : s_test_z,
                       inputs[1] : s_test_xdata,
                       inputs[2] : s_test_xdata_mask}
        if offset_eval is not None:
            givens_eval = {inputs[0] : s_eval_z,
                           inputs[1] : s_eval_xdata,
                           inputs[2] : s_eval_xdata_mask}
    else:
        givens = {inputs[0] : s_train_z[:, n_batch * index : n_batch * (index + 1), :],
                  inputs[1] : s_train_xdata[:, n_batch * index : n_batch * (index + 1), :]}
        givens_test = {inputs[0] : s_test_z,
                       inputs[1] : s_test_xdata}
        if offset_eval is not None:
            givens_eval = {inputs[0] : s_eval_z,
                           inputs[1] : s_eval_xdata}

    # load parameters from the specified initfile, if it exists
    if initfile is not None and os.path.isfile(initfile):
        print "Using file %s to initialize parameters" % initfile
        L=cPickle.load(file(initfile,'r'))
        best_params_load=L['best_params']
        V=theano.shared(best_params_load[0])
        U=theano.shared(best_params_load[1])
        hidden_bias=theano.shared(best_params_load[2])
       
        if (model=='LSTM'):
            for iparam in range(len(best_params_load)):
                dupdate={parameters[iparam] : theano.shared(best_params_load[iparam])}
                givens.update(dupdate)
                givens_test.update(dupdate)
                if offset_eval is not None:
                    givens_eval.update(dupdate)
        elif (Wimpl=='adhoc'):
            reflection=theano.shared(best_params_load[3])
            out_bias=theano.shared(best_params_load[4])
            theta=theano.shared(best_params_load[5])
            h_0=theano.shared(best_params_load[6])

            if ( ('timit' in dataset) and flag_generator):
                
                if (n_layers==1):
                    print "Dataset is timit and we are running a generator with 1 layer, so initialize U and c from initfile."
                    print ""
                    # only use U and out_bias for initialization
                    givens.update({parameters[1] : U,
                                   parameters[4] : out_bias})
                    givens_test.update({parameters[1] : U,
                                        parameters[4] : out_bias})
                    if offset_eval is not None:
                        givens_eval.update({parameters[1] : U,
                                            parameters[4] : out_bias})
            else:
                givens_test.update({parameters[0] : V,
                                    parameters[1] : U,
                                    parameters[2] : hidden_bias,
                                    parameters[3] : reflection,
                                    parameters[4] : out_bias,
                                    parameters[5] : theta,
                                    parameters[6] : h_0})
                if offset_eval is not None:
                    givens_eval.update({parameters[0] : V,
                                        parameters[1] : U,
                                        parameters[2] : hidden_bias,
                                        parameters[3] : reflection,
                                        parameters[4] : out_bias,
                                        parameters[5] : theta,
                                        parameters[6] : h_0})

        elif (Wimpl=='full'):
            out_bias=theano.shared(best_params_load[3])
            h_0=theano.shared(best_params_load[4])
            Waug=theano.shared(best_params_load[5])

            if (('timit' in dataset) and flag_generator):
                
                if (n_layers==1):
                    print "Dataset is timit and we are running a generator with 1 layer, so initialize U and c from initfile."
                    print ""
                    # only use U and out_bias for initialization
                    givens.update({parameters[1] : U,
                                   parameters[3] : out_bias})
                    givens_test.update({parameters[1] : U,
                                        parameters[3] : out_bias})
                    if offset_eval is not None:
                        givens_eval.update({parameters[1] : U,
                                            parameters[3] : out_bias})
            else:
                givens_test.update({parameters[0] : V,
                                    parameters[1] : U,
                                    parameters[2] : hidden_bias,
                                    parameters[3] : out_bias,
                                    parameters[4] : h_0,
                                    parameters[5] : Waug})
                if offset_eval is not None:
                    givens_eval.update({parameters[0] : V,
                                        parameters[1] : U,
                                        parameters[2] : hidden_bias,
                                        parameters[3] : out_bias,
                                        parameters[4] : h_0,
                                        parameters[5] : Waug})


    if (dataset == 'synthgen') and flag_onlyOptimW:
        # we are only optimizing W, so use some ground-truth parameters from
        # the synth networks
        
        givens[parameters[0]] = Vaug
        givens[parameters[1]] = Uaug
        givens[parameters[2]] = theano.shared(parameters_synth[2].get_value()) # hidden_bias
        
        # out_bias
        if (Wimpl == 'adhoc'):
            givens[parameters[4]] = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX))
        elif (Wimpl == 'givens') or (Wimpl == 'full'):
            givens[parameters[3]] = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX))
       
        # h_0
        if (Wimpl == 'adhoc'):
            givens[parameters[6]] = h_0_synth
        elif (Wimpl == 'givens') or (Wimpl == 'full'):
            givens[parameters[4]] = h_0_synth
        
        givens_test[parameters[0]] = Vaug
        givens_test[parameters[1]] = Uaug
        givens_test[parameters[2]] = theano.shared(parameters_synth[2].get_value()) # hidden_bias
        
        # out_bias
        if (Wimpl == 'adhoc'):
            givens_test[parameters[4]] = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX))
        elif (Wimpl == 'givens') or (Wimpl == 'full'):
            givens_test[parameters[3]] = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX))

        # h_0
        if (Wimpl == 'adhoc'):
            givens_test[parameters[6]] = h_0_synth
        elif (Wimpl == 'givens') or (Wimpl == 'full'):
            givens_test[parameters[4]] = h_0_synth

        if offset_eval is not None:
            givens_eval[parameters[0]] = Vaug
            givens_eval[parameters[1]] = Uaug
            givens_eval[parameters[2]] = theano.shared(parameters_synth[2].get_value()) # hidden_bias
            
            # out_bias
            if (Wimpl == 'adhoc'):
                givens_eval[parameters[4]] = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX))
            elif (Wimpl == 'givens') or (Wimpl == 'full'):
                givens_eval[parameters[3]] = theano.shared(np.zeros((2*n_output,), dtype=theano.config.floatX))

            # h_0
            if (Wimpl == 'adhoc'):
                givens_eval[parameters[6]] = h_0_synth
            elif (Wimpl == 'givens') or (Wimpl == 'full'):
                givens_eval[parameters[4]] = h_0_synth

    
    train = theano.function([index], [costs[0],costs[1]], givens=givens, updates=updates)
    test = theano.function([], [costs[0], costs[1], costs[2], costs[3], costs[4], costs[5]], givens=givens_test)
    if offset_eval is not None:
        evalf = theano.function([], [costs[0], costs[1], costs[2], costs[3], costs[4], costs[5]], givens=givens_eval)

    # --- Training Loop ---------------------------------------------------------------

    train_loss = []
    train_ref = []
    if (loss_function=='MSEplusL1'):
        train_mse = []
        test_mse = []
    train_time= []
    test_loss = []
    test_ref = []
    test_time = []
    best_params = [p.get_value() for p in parameters]
    best_rms = [r.get_value() for r in rmsprop]
    best_test_loss = 1e10
    #num_allowed_test_inc=10
    num_test_inc=0
    shuffle_rng=np.random.RandomState(314)
    data_xdata = s_train_xdata.get_value()
    for i in xrange(n_iter):
        if (i % num_batches == 0):
            # reshuffle batch indices
            inds = shuffle_rng.permutation(n_train)
            data_z = s_train_z.get_value()
            s_train_z.set_value(data_z[:,inds,:])
            data_xdata = s_train_xdata.get_value()
            s_train_xdata.set_value(data_xdata[:,inds,:])
            if flag_use_mask:
                data_xdata_mask = s_train_xdata_mask.get_value()
                s_train_xdata_mask.set_value(data_xdata_mask[:,inds,:])

        start_time=time.time()
        mse, extra = train(i % num_batches)
        elapsed_time = time.time() - start_time
        train_loss.append(mse)
        msp = (data_xdata[:, n_batch * (i%num_batches):n_batch * (i%num_batches+1),:]**2).mean() #mean-squared power of reference
        train_ref.append(msp)
        train_time.append(elapsed_time)
        print "Iteration:", i
        if (loss_function=='MSEplusL1'):
            train_mse.append(extra)
            print "MSE + L1: ", mse
            print "MSE     : ", extra
            print "NMSE    : ", extra/msp
        else:
            print "MSE: ", mse
            print "NMSE:", mse/msp
        print "Time:", elapsed_time
        print

        if (i % iters_per_validCheck==0):
            start_time=time.time()
            mse, extra, xgen, ht, nmse_local, cost_steps = test()
            elapsed_time = time.time() - start_time
            msp = (test_xdata**2).mean()
            print
            print "TEST"
            if (loss_function=='MSEplusL1'):  
                test_mse.append(extra)
                print "MSE + L1: ", mse
                print "MSE     : ", extra
                print "NMSE    : ", extra/msp
            else:
                print "MSE: ", mse
                print "NMSE global:", mse/msp
            print "NMSE local:", nmse_local.mean()
            print "Time:", elapsed_time
            print
            test_loss.append(mse)
            test_ref.append(msp)
            test_time.append(elapsed_time)

            if mse < best_test_loss:
                best_params = [p.get_value() for p in parameters]
                best_rms = [r.get_value() for r in rmsprop]
                best_test_loss = mse
                best_xgen = xgen
                best_ht = ht
                best_nmse_local = nmse_local
            else:
                num_test_inc=num_test_inc+1
                print "No improvement in test loss, %d of %d allowed" % (num_test_inc,num_allowed_test_inc)
                print ""
                if num_test_inc==num_allowed_test_inc:
                    print "Number of allowed test loss increments reached. Returning..."
                    print ""
                    return

            save_vals = {'parameters': [p.get_value() for p in parameters],
                         'rmsprop': [r.get_value() for r in rmsprop],
                         'train_loss': train_loss,
                         'train_ref': train_ref,
                         'train_time': train_time,
                         'test_loss': test_loss,
                         'test_ref': test_ref,
                         'test_time': test_time,
                         'best_params': best_params,
                         'best_rms': best_rms,
                         'best_test_loss': best_test_loss,
                         'best_xgen': best_xgen,
                         #'best_ht': best_ht,
                         'best_nmse_local': best_nmse_local,
                         'model': model,
                         'stats': stats}

            if (loss_function=='MSEplusL1'):
                save_vals['train_mse']=train_mse
                save_vals['test_mse']=test_mse

            cPickle.dump(save_vals,
                         file(savefile, 'wb'),
                         cPickle.HIGHEST_PROTOCOL)

    # run evaluation data
    if offset_eval is not None:
        print ""
        print "Running forward model on evaluation data using best validation parameters..."
        start_time=time.time()
        mse, extra, xgen, ht, nmse_local, cost_steps = evalf()
        elapsed_time = time.time() - start_time

        print "Forward pass took %f seconds" % elapsed_time
        
        msp = (eval_xdata**2).mean()

        print "MSE=%f, ref=%f, NMSE=%f" % (mse,msp,mse/msp)

        save_vals = {'parameters': [p.get_value() for p in parameters],
                     'eval_loss': mse,
                     'eval_ref': msp,
                     'eval_time': elapsed_time,
                     'best_params': best_params,
                     'xgen': xgen,
                     'ht': ht,
                     'model': model,
                     'stats': stats}

        cPickle.dump(save_vals,
                     file(savefile+'_eval', 'wb'),
                     cPickle.HIGHEST_PROTOCOL)

    if outdir is not None: 
        outdir=''.join(["/data1/prediction_audio_out/", outdir])
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        if not outdir[-1]=='/':
            outdir=outdir + '/' #append a slash on the end
        #write out synthetized audio
        mse, extra, xgen, ht, nmse_local, cost_steps = test()

        cPickle.dump({'ht': ht},
                     file('ht_cur', 'wb'),
                     cPickle.HIGHEST_PROTOCOL)

        Yest=xgen
        # Yest is shape nframMax x nutt x 2*nch*F
        nutt=Yest.shape[1]
        nsrc=1
        for iutt in range(nutt):
            #perform inverse STFT for estimates Yest
            Yest_cur = np.transpose(np.squeeze(Yest[:,iutt,:]),(1,0))
            # Yest_cur is 2*nch*F x nframMax
            yestr = iAugSTFT(Yest_cur,F,1,flag_unwrap_phase,flag_noDiv=flag_noDiv,window=params_stft['window'],hop=params_stft['hop'])
            # yestr is 1 x nsampl x nch

            #perform inverse STFT for references test_y
            test_xdata_cur = np.transpose(np.squeeze(test_xdata[:,iutt,:]),(1,0))
            # test_y_cur is 2*nch*F x nframMax
            yr = iAugSTFT(test_xdata_cur,F,1,flag_unwrap_phase,flag_noDiv=flag_noDiv,window=params_stft['window'],hop=params_stft['hop'])
            # yr is 1 x nsampl x nch

            #build parts of output wavfile
            wavfile_in_cur=wavfiles_test[iutt]
            wavfile_in_split=wavfile_in_cur.split('/')
            filename=wavfile_in_split[-2] + '_' + wavfile_in_split[-1]

            print 'Writing out wav for file %d of %d total...' % (iutt+1,nutt)

            #write out audio of data example (to check the istft plumbing)
            filename_out=filename
            path_out=''.join((outdir,filename_out))
            wavwrite(path_out,8e3,np.squeeze(yr[0,:,:])) 
            #write out audio file of reconstructed output
            path_out=''.join((outdir,filename_out.replace('.wav','') + '_gen%d' % iutt, '.wav'))
            wavwrite(path_out,8e3,np.squeeze(yestr[0,:,:]))

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="training a model")
    parser.add_argument("n_iter", type=int, default=20000)
    parser.add_argument("n_batch", type=int, default=20)
    parser.add_argument("n_hidden", type=int, default=512)
    parser.add_argument("learning_rate", type=float, default=0.001)
    parser.add_argument("savefile")
    parser.add_argument("model", default='complex_RNN')
    parser.add_argument("input_type", default='categorical')
    parser.add_argument("out_every_t", default='False')
    parser.add_argument("loss_function", default='MSE')
    parser.add_argument("fold",default='fold1')
    parser.add_argument("scene")
    parser.add_argument("--n_reflections", default=8, help="number of reflections for CUE-RNN")
    parser.add_argument("--flag_telescope", default=True, help="whether to use telescoping reflections (True) or full reflections (False)")
    parser.add_argument("--nch", default=1, help="how many channels of audio input to use (default=1)")
    parser.add_argument("--flag_unwrap_phase", default=True, help="remove window hop phase on target STFTs (default=True)")
    parser.add_argument("--indir", default="audio_8khz", help="input directory for DCASE2016 audio file clips (default=audio_8khz)")
    parser.add_argument("--outdir", default=None, help="output directory for reconstructed files (default=None)")
    parser.add_argument("--dataset", default="timit", help="dataset to use (default=timit)")
    parser.add_argument("--initfile", default=None, help="savefile to initialize generator network with (default=None)")
    parser.add_argument("--flag_feed_forward", default=True, help="disable recurrent connections (default=True)")
    parser.add_argument("--flag_generator", default=False, help="run as a generator i.e., inputs are random unit-variance circular complex Gaussian vectors  (default=False)")
    parser.add_argument("--downsample_train", default=1, help="downsampling factor for training data  (default=1)")
    parser.add_argument("--downsample_test", default=1, help="downsampling factor for test data  (default=1)")

    parser.add_argument("--time_steps", default=0, help="number of time steps for sequences when using dataset='synthgen' (default=0, not used if dataset does not equal 'synthgen')")
    parser.add_argument("--n_Givens", default=0, help="number of Givens rotations to use for uRNN (default=0; if set to None or 0, ad-hoc parameterization of [Arjovsky, Shah, and Bengio 2015] is used)")
    parser.add_argument("--prng_seed_Givens", default=52016, help="seed to initialize random number generator that determines which indices will be used in Givens parameterization (default=52016)")
    parser.add_argument("--num_allowed_test_inc", default=10, help="number of allowed increases in the test error before stopping training (default=10)")
    parser.add_argument("--iters_per_validCheck", default=20, help="number of training iterations between validation checks on test set (default=20)")
    parser.add_argument("--flag_useFullW", default=False, help="use a full unitary matrix for W, using Stiefel manifold projected gradient for optimization; overrides n_Givens (default=False)")
    parser.add_argument("--flag_onlyOptimW", default=True, help="if dataset is synthgen, only optimize W, and not the other parameters V, U, hidden_bias, out_bias (default=True)")
    parser.add_argument("--lam", default=0.0, help="used as L1 regularization weight if loss_function is MSEplusL1 (default=0.0)")
    parser.add_argument("--Vnorm", default=0.0, help="normalize rows of V to equal this number (default=0.0, which means no normalization is performed)")
    parser.add_argument("--Unorm", default=0.0, help="normalize columnss of U to equal this number (default=0.0, which means no normalization is performed)")
    parser.add_argument("--n_layers", default=1, help="number of RNN layers to use (default=1)")
    parser.add_argument("--num_pred_steps", default=0, help="number of steps to predict ahead (default=0)")
    parser.add_argument("--hidden_bias_mean", default=0.1, help="mean of initial hidden bias in all layers (default=0.1)")
    parser.add_argument("--data_transform", default="", help="apply a transformation to the input and output data (default='', options: 'logmag')")
    parser.add_argument("--bwe_frac", default=1.0, help="if less than 1.0, sets up data to do bandwidth expansion by using the lower (1-bwe_frac) portion of the data to predict the upper bwe_frac portion of the data (default=1.0)")
    parser.add_argument("--data_normalization", default="none", help="compute statistics and normalize training and test data per feature dimension (default='none', options: any combination of lower-case 'mean', 'var', and/or 'perUtt'. Defaults to global statistics (i.e., computed over all time steps and sequences))")
    parser.add_argument("--offset_eval", default=-1, help="starting index of eval set, uses downsample_test (default=None)")
    parser.add_argument("--olap", default=50, help="overlap of STFT window, as number between 1 and 99. Only achieves perfect reconstruction for certain values, dependent on STFT window choice (e.g., 'sqrt_hann' with olap=50 gives PR) (default=50)")
    parser.add_argument("--window", default='hann', help="STFT window, provided as a string. Options: 'hann', 'sqrt_hann' (default='hann')")
    parser.add_argument("--flag_noDiv", default=0, help="if set, does not divide reconstructed time series by overlap-added squared window (default=0)")
    parser.add_argument("--flag_noComplexConstraint", default=0, help="if set, relaxes complex constraint on input transform V and output transform U for complex_RNNs (default=0)")
    parser.add_argument("--Winit", default="svd", help="Initialization method of synth_W in synthgen experiments. Options: 'svd', 'adhoc', 'adhoc2x' (default='svd')")
    parser.add_argument("--seed", default=1234, help="random seed for LSTM and complex_RNN (default=1234)")
    parser.add_argument("--optim_alg", default="rmsprop", help="optimization algorithm (default='rmsprop')")
    parser.add_argument("--n_utt_eval_spec", default=-1, help="number of evaluation utterances, from offset_eval (default=-1)")

    args = parser.parse_args()
    dict = vars(args)

    kwargs = {'n_iter': dict['n_iter'],
              'n_batch': dict['n_batch'],
              'n_hidden': dict['n_hidden'],
              'learning_rate': np.float32(dict['learning_rate']),
              'savefile': dict['savefile'],
              'model': dict['model'],
              'input_type': dict['input_type'],
              'out_every_t': 'True'==dict['out_every_t'],
              'loss_function': dict['loss_function'],
              'fold': dict['fold'],
              'scene': dict['scene'],
              'n_reflections': int(args.n_reflections),
              'flag_telescope': bool(np.int(args.flag_telescope)),
              'nch': int(args.nch),
              'flag_unwrap_phase': bool(np.int(args.flag_unwrap_phase)),
              'indir': dict['indir'],
              'outdir': dict['outdir'],
              'dataset': dict['dataset'],
              'initfile': dict['initfile'],
              'flag_feed_forward': bool(np.int(args.flag_feed_forward)),
              'flag_generator': bool(np.int(args.flag_generator)),
              'downsample_train': int(args.downsample_train),
              'downsample_test': int(args.downsample_test),
              'time_steps': int(args.time_steps),
              'n_Givens': int(args.n_Givens),
              'prng_seed_Givens': int(args.prng_seed_Givens),
              'num_allowed_test_inc': int(args.num_allowed_test_inc),
              'iters_per_validCheck': int(args.iters_per_validCheck),
              'flag_useFullW': bool(np.int(args.flag_useFullW)),
              'flag_onlyOptimW': bool(np.int(args.flag_onlyOptimW)),
              'lam': np.float32(dict['lam']),
              'Vnorm': np.float32(dict['Vnorm']),
              'Unorm': np.float32(dict['Unorm']),
              'n_layers': int(args.n_layers),
              'num_pred_steps': int(args.num_pred_steps),
              'hidden_bias_mean': np.float32(dict['hidden_bias_mean']),
              'data_transform': dict['data_transform'],
              'bwe_frac': np.float32(dict['bwe_frac']),
              'data_normalization': dict['data_normalization'],
              'offset_eval': int(args.offset_eval),
              'olap': np.float32(dict['olap']),
              'window': dict['window'],
              'flag_noDiv': bool(np.int(args.flag_noDiv)),
              'flag_noComplexConstraint': bool(np.int(args.flag_noComplexConstraint)),
              'Winit': dict['Winit'],
              'seed': int(args.seed),
              'optim_alg': dict['optim_alg'],
              'n_utt_eval_spec': int(args.n_utt_eval_spec)}


    main(**kwargs)
