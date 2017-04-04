import sys
sys.setrecursionlimit(10000)
import cPickle
import gzip
import pdb
import numpy as np
import argparse, timeit
import os
import scipy
import scipy.signal
import scipy.io.wavfile
import librosa
import six
import librosa.util as util
import scipy.fftpack as fft

def wavread(wavfile):
    fs,x=scipy.io.wavfile.read(wavfile) #x will be nsampl x nch
    x=np.transpose(x).astype(np.float32) #convert x to float32, transpose to nch x nsampl
    x=x/32768.0
    return x

def wavwrite(wavfile,fs,x):
    # x should be nsampl x nch
    if x.dtype==np.float32:
        #convert float32 data to int16
        xMaxAbs=np.max(np.abs(x))
        if xMaxAbs>1:
            x=x/xMaxAbs
        x=np.int16(x*32767.0)
    scipy.io.wavfile.write(wavfile,fs,x.T)


def istft_noDiv(stft_matrix, hop_length=None, win_length=None, window=None,
          center=True, dtype=np.float32):
    """

    #Copied from librosa's spectrum.py file, removing division by squared
    window, which shouldn't be necessary and can cause problems in recon.
    
    Inverse short-time Fourier transform (ISTFT).
    Converts a complex-valued spectrogram `stft_matrix` to time-series `y`
    by minimizing the mean squared error between `stft_matrix` and STFT of
    `y` as described in [1]_.
    In general, window function, hop length and other parameters should be same
    as in stft, which mostly leads to perfect reconstruction of a signal from
    unmodified `stft_matrix`.
    Parameters
    ----------
    stft_matrix : np.ndarray [shape=(1 + n_fft/2, t)]
        STFT matrix from `stft`
    hop_length  : int > 0 [scalar]
        Number of frames between STFT columns.
        If unspecified, defaults to `win_length / 4`.
    win_length  : int <= n_fft = 2 * (stft_matrix.shape[0] - 1)
        When reconstructing the time series, each frame is windowed
        and each sample is normalized by the sum of squared window
        according to the `window` function (see below).
        If unspecified, defaults to `n_fft`.
    window      : None, function, np.ndarray [shape=(n_fft,)]
        - None (default): use an asymmetric Hann window
        - a window function, such as `scipy.signal.hanning`
        - a user-specified window vector of length `n_fft`
    center      : boolean
        - If `True`, `D` is assumed to have centered frames.
        - If `False`, `D` is assumed to have left-aligned frames.
    dtype       : numeric type
        Real numeric type for `y`.  Default is 32-bit float.
    Returns
    -------
    y : np.ndarray [shape=(n,)]
        time domain signal reconstructed from `stft_matrix`
    Raises
    ------
    ParameterError
        If `window` is supplied as a vector of length `n_fft`
    See Also
    --------
    stft : Short-time Fourier Transform
    Examples
    --------
    >>> y, sr = librosa.load(librosa.util.example_audio_file())
    >>> D = librosa.stft(y)
    >>> y_hat = librosa.istft(D)
    >>> y_hat
    array([ -4.812e-06,  -4.267e-06, ...,   6.271e-06,   2.827e-07], dtype=float32)
    Exactly preserving length of the input signal requires explicit padding.
    Otherwise, a partial frame at the end of `y` will not be represented.
    >>> n = len(y)
    >>> n_fft = 2048
    >>> y_pad = librosa.util.fix_length(y, n + n_fft // 2)
    >>> D = librosa.stft(y_pad, n_fft=n_fft)
    >>> y_out = librosa.util.fix_length(librosa.istft(D), n)
    >>> np.max(np.abs(y - y_out))
    1.4901161e-07
    """

    n_fft = 2 * (stft_matrix.shape[0] - 1)

    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length / 4)

    if window is None:
        # Default is an asymmetric Hann window.
        ifft_window = scipy.signal.hann(win_length, sym=False)

    elif six.callable(window):
        # User supplied a windowing function
        ifft_window = window(win_length)

    else:
        # User supplied a window vector.
        # Make it into an array
        ifft_window = np.asarray(window)

        # Verify that the shape matches
        if ifft_window.size != n_fft:
            raise ParameterError('Size mismatch between n_fft and window size')

    # Pad out to match n_fft
    ifft_window = util.pad_center(ifft_window, n_fft)

    # scale the window
    ifft_window = ifft_window*(2.0/(win_length/hop_length))

    n_frames = stft_matrix.shape[1]
    expected_signal_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_sum = np.zeros(expected_signal_len, dtype=dtype)
    ifft_window_square = ifft_window * ifft_window

    for i in range(n_frames):
        sample = i * hop_length
        spec = stft_matrix[:, i].flatten()
        spec = np.concatenate((spec.conj(), spec[-2:0:-1]), 0)
        ytmp = ifft_window * fft.ifft(spec).real

        y[sample:(sample + n_fft)] = y[sample:(sample + n_fft)] + ytmp
        # shouldn't need to do this sum of the squared window:
        #ifft_window_sum[sample:(sample + n_fft)] += ifft_window_square

    # don't do this:
    ## Normalize by sum of squared window
    #approx_nonzero_indices = ifft_window_sum > util.SMALL_FLOAT
    #y[approx_nonzero_indices] /= ifft_window_sum[approx_nonzero_indices]

    if center:
        y = y[int(n_fft // 2):-int(n_fft // 2)]

    return y

def stft_mc(x,N=1024,hop=None,window='hann'):
    # N=1024
    if hop is None:
        hop=N/2
    S=x.shape
    if len(S)==1:
        nch=1
        nsampl=len(x)
        x=np.reshape(x,(1,nsampl)) 
    else:
        nch=S[0]
        nsampl=S[1]
    xdtype=x.dtype
    nfram=int(scipy.ceil(float(nsampl)/float(hop)))
    npad=int(nfram)*hop-nsampl
    pad=np.zeros((nch,npad)).astype(xdtype)
    x=np.concatenate((x,pad),axis=1)
    #pad the edges to avoid window taper effects
    pad=np.zeros((nch,N)).astype(xdtype)
    x=np.concatenate((pad,x,pad),axis=1)
    for ich in range(0,nch):
        x0=x[ich,:]
        if not x0.flags.c_contiguous:
            x0=x0.copy(order='C')
        X0=librosa.core.stft(x0,n_fft=N,hop_length=hop,window=window,center=False,dtype=np.complex64)
        if ich==0:
            X=np.zeros((N/2+1,X0.shape[1],nch)).astype(np.complex64)
            X[:,:,0]=X0
        else:
            X[:,:,ich]=X0
    return X

def istft_mc(X,hop,dtype=np.float32,nsampl=None,flag_noDiv=0,window=None):
    #assumes X is of shape F x nfram x nch, where F=Nwin/2+1
    #returns xr of shape nch x nsampl
    N=2*(X.shape[0]-1)
    nch=X.shape[2]
    for ich in range(0,nch):
        X0=X[:,:,ich]
        if flag_noDiv:
            x0r=istft_noDiv(X0,hop_length=hop,center=False,window=window,dtype=dtype)
        else:
            x0r=librosa.core.istft(X0,hop_length=hop,center=False,window=window,dtype=dtype)
        if ich==0:
            xr=np.zeros((nch,len(x0r))).astype(dtype)
            xr[0,:]=x0r
        else:
            xr[ich,:]=x0r
    #trim off extra zeros
    nfram=xr.shape[1]
    xr=xr[:,0:(nfram-N)]
    nfram=xr.shape[1]
    xr=xr[:,N:]
    if not nsampl is None:
        xr=xr[:,0:nsampl]
    return xr, N

def AugSTFT(x,N,hop,flag_unwrap_phase,window=None):
    F=N/2+1
    Xf=stft_mc(x,N,hop=hop,window=window)
    Xf=Xf[:,:,0] # take first channel
    nfram=Xf.shape[1]
    if flag_unwrap_phase:
        # remove window hop phases:
        Xphase=np.float32(np.unwrap(np.angle(Xf),axis=1))
        frange=np.arange(0,F,dtype=np.float32)/N
        trange=np.arange(0,nfram,dtype=np.float32)*hop
        Xphase=Xphase-2*np.pi*np.outer(frange,trange)
        Xf=np.abs(Xf)*np.exp(1j*Xphase)
    Xaug=np.concatenate((np.real(Xf),np.imag(Xf)),axis=0)
    return Xaug

def iAugSTFT(X,F,nsrc,flag_unwrap_phase,flag_noDiv=0,window='hann',hop=None):
    #reconstructs a time series from augmented STFT
    #assumes the augmented STFT X is of shape 2*nsrc*nch*F x nfram
    #returns xr of shape nsrc x nsampl x nch
    Nwin=2*(F-1)
    if hop is None:
        hop=Nwin/2
    n_tot = X.shape[0]
    nfram = X.shape[1]
    n_reim = n_tot/2
    # convert X to complex
    Xc = X[:n_reim,:] + 1j*X[n_reim:,:]
    nch = Xc.shape[0]/(nsrc*F)
    for isrc in range(nsrc):
        Xc_src = Xc[(isrc*nch*F):((isrc+1)*nch*F),:]
        #Xc_src is nch*F x nfram
        Xc_cur = np.reshape(Xc_src,(F,nch,nfram),order='F')
        #Xc_cur is now F x nch x nfram
        Xc_cur = np.transpose(Xc_cur,(0,2,1))
        #Xc_cur is now F x nfram x nch

        if flag_unwrap_phase:
            # add window hop phases:
            Xphase=np.float32(np.unwrap(np.angle(Xc_cur),axis=1))
            frange=np.arange(0,F,dtype=np.float32)/Nwin
            trange=np.arange(0,nfram,dtype=np.float32)*np.float32(hop)
            Xphase=Xphase+2*np.pi*np.reshape(np.outer(frange,trange),[F,nfram,1])
            Xc_cur=np.abs(Xc_cur)*np.exp(1j*Xphase)

        xr_cur = istft_mc(Xc_cur,hop,flag_noDiv=flag_noDiv,window=window)
        xr_cur = xr_cur[0]
        #xr_cur is nch x nsampl
        nsampl = xr_cur.shape[1]
        if isrc==0:
            xr = np.zeros((nsrc,nsampl,nch),dtype=np.float32)
        xr[isrc,:,:]=np.transpose(xr_cur,(1,0))
    return xr

def load_wavfiles_names(path):
    wavfiles=list()
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(('_mix1.wav','_mix3.wav')):
                wavfile_noisy=os.path.join(root,file)
                utt=file.split('_')
                wavfile_s1=os.path.join(root,''.join((utt[0],'_src1.wav')))
                wavfile_s1n=os.path.join(root,''.join((utt[0],'_src1n.wav')))
                wavfiles.append((wavfile_noisy,wavfile_s1,wavfile_s1n))

    return wavfiles

def generate_data(wavfiles,params_stft):
    N=params_stft['N']
    hop=params_stft['hop']
    nch=params_stft['nch']
    F=N/2+1

    # initialize matrices to hold concatenated STFTs
    X=np.zeros((nch*F,0)).astype(np.complex64)
    S1=np.zeros((nch*F,0)).astype(np.complex64)
    S2=np.zeros((nch*F,0)).astype(np.complex64)

    # initialize frame indices for individual files
    fidx=np.zeros((len(wavfiles),2)).astype(np.int32)
    ifidx=0
    ifile=0
    for wavfile in wavfiles:
        print "Read file %d of %d total: %s" % (ifile+1,len(wavfiles),wavfile[0])
        # read in noisy mixture
        x=wavread(wavfile[0])
        Xcur=stft_mc(x,N,hop)
        Xcur=Xcur[:,:,:nch] #restrict to desired number of channels
        Xcur=np.transpose(Xcur,(0,2,1)) #is now F x nch x nfram
        Xcur=np.reshape(Xcur,(nch*F,Xcur.shape[2]),order='F')
        # read in source 1 image
        s1=wavread(wavfile[1])
        S1cur=stft_mc(s1,N,hop)
        S1cur=S1cur[:,:,:nch] #restrict to desired number of channels
        S1cur=np.transpose(S1cur,(0,2,1)) #is now F x nch x nfram
        S1cur=np.reshape(S1cur,(nch*F,S1cur.shape[2]),order='F')
        # read in source 1 image plus noise
        s1n=wavread(wavfile[2])
        s2=x-s1n
        S2cur=stft_mc(s2,N,hop)
        S2cur=S2cur[:,:,:nch] #restrict to desired number of channels
        S2cur=np.transpose(S2cur,(0,2,1)) #is now F x nch x nfram
        S2cur=np.reshape(S2cur,(nch*F,S2cur.shape[2]),order='F')
        # update frame indices for this file
        nfram=Xcur.shape[1]
        fidx[ifile,0]=ifidx
        ifidx+=nfram
        fidx[ifile,1]=ifidx
        ifile+=1
        X=np.concatenate((X,Xcur),axis=1)
        S1=np.concatenate((S1,S1cur),axis=1)
        S2=np.concatenate((S2,S2cur),axis=1)
    Xaug=np.concatenate((np.real(X),np.imag(X)),axis=0)
    Y=np.concatenate((S1,S2),axis=0)
    Yaug=np.concatenate((np.real(Y),np.imag(Y)),axis=0)
    return Xaug,Yaug,fidx

