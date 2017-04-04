#!/usr/bin/python

import os
import sys, getopt
import util
import cPickle
import numpy as np

# Print iterations progress
def printProgress (iteration, total, prefix = '', suffix = '', decimals = 2, barLength = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
    """
    filledLength    = int(round(barLength * iteration / float(total)))
    percents        = round(100.00 * (iteration / float(total)), decimals)
    bar             = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('%s [%s] %s%s %s\r' % (prefix, bar, percents, '%', suffix)),
    sys.stdout.flush()
    if iteration == total:
        print("\n")

def load_TIMIT_test_xdata(savefile):
    n_output=129
    if ('trainNoSA' in savefile):
        data=cPickle.load( open("timit_data_trainNoSA_dev_coreTest"))
        fidx_test=np.asarray(data['fidx_test'])
        fidx_test=fidx_test[400:400+192,:]
        test_xdata_stack=np.asarray(data['test_xdata_stack']).astype(np.float32)
    else:
        data=cPickle.load( open("timit_data"))
        fidx_test=np.asarray(data['fidx_test'])
        test_xdata_stack=np.asarray(data['test_xdata_stack']).astype(np.float32)
    lens_test=fidx_test[:,1]-fidx_test[:,0]
    n_framMax_test=np.max(lens_test)
    n_utt_test=len(lens_test)
    test_xdata = np.zeros((n_framMax_test,2*n_output,n_utt_test)).astype(np.float32)
    test_mask = np.zeros_like(test_xdata).astype(np.float32)
    for iutt in range(n_utt_test):
        test_mask[:lens_test[iutt],:,iutt]=1.0
        test_xdata[:lens_test[iutt],:,iutt]=np.transpose(test_xdata_stack[:,fidx_test[iutt,0]:fidx_test[iutt,1]])
    # test_xdata is in augmented form and is now of dimensions n_framMax_test x 2*n_output x n_utt_test

    # to get scan to work properly, transpose x and y to be of size n_framMax x n_utt x n_<input,output>
    test_mask=np.transpose(test_mask,[0,2,1])
    test_xdata =np.transpose(test_xdata,[0,2,1])
    print "Loaded TIMIT test data"
    return test_xdata, test_mask


def main(argv):
    savefile = ''
    outputfolder = ''
    try:
        opts, args = getopt.getopt(argv,"hs:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'recon_timitpred.py -s <savefile> -o <output folder>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'recon_timitpred.py -s <savefile> -o <output folder>'
            sys.exit()
        elif opt in ("-s"):
            savefile = arg
        elif opt in ("-o"):
            outputfolder = arg
    print 'Savefile is ', savefile
    print 'Output folder is ', outputfolder

    # load reference data
    print "Loading TIMIT test data..."
    test_xdata,test_mask=load_TIMIT_test_xdata(savefile)
    if ('trainNoSA' not in savefile):
        test_mask=test_mask[:,1::2,:]
        test_xdata=test_xdata[:,1::2,:]

    # load results file that contains predicted STFT log-magnitudes
    results_eval=cPickle.load( open(savefile, "rb"))
    best_xgen=np.asarray(results_eval['xgen'])
    best_xgen=best_xgen[:test_xdata.shape[0],:,:]
    best_test_loss=np.asarray(results_eval['eval_loss'])

    # undo data normalization
    normalize_str=''
    if ('_normalizeMeanVarGlobal' in savefile):
        normalize_str='_normalizeMeanVarGlobal'
    elif ('_normalizeVarGlobal' in savefile):
        normalize_str='_normalizeVarGlobal' 
    if ('Var' in normalize_str):
        stats=results_eval['stats']
        stats_cur=stats['eval_xdata_stats']
        best_xgen_std=stats_cur['std']
        best_xgen=best_xgen*(np.float32(1e-7)+np.float32(np.sqrt(2))*np.tile(best_xgen_std,(1,1,2))) 
    if ('Mean' in normalize_str):
        stats=results_eval['stats']
        stats_cur=stats['eval_xdata_stats']
        best_xgen_mean=stats_cur['mean']
        best_xgen=best_xgen+best_xgen_mean
                                              
    # build complex-valued STFTs of reference and predicted
    npred=1
    n_input=129
    n_output=129
    test_xdata_logmag=10.0*np.log10(1e-5 + test_xdata[:,:,:129]**2 + test_xdata[:,:,129:]**2)
    test_xdata_c=test_xdata[:,:,:129]+np.complex64(1j)*test_xdata[:,:,129:]
    test_xdata_a=np.concatenate( [np.real(test_xdata_c),np.imag(test_xdata_c)],axis=2)
    
    magsq=test_mask[:,:,0:1]*((10**( best_xgen/10.0 )))
    best_xgen_mag=np.sqrt( magsq )
    test_xdata_mag=np.sqrt( test_mask[:,:,0:1]*((10**( test_xdata_logmag/10.0 ))) )
    best_xgen_c=best_xgen_mag
    best_xgen_c=best_xgen_c[:-npred,:,:]*np.exp(np.complex64(1j)*np.angle(test_xdata_c[npred:,:,:]))
    best_xgen_complete=np.concatenate( [np.real(best_xgen_c),np.imag(best_xgen_c)],axis=2)
   
    n_utt=best_xgen.shape[1]
    print "Reconstructing audio..."
    for uidx in range(n_utt):
        #printProgress(uidx+1, n_utt, prefix = 'Progress:', suffix = 'Complete', barLength = 50)
        Tcur=int(np.sum(test_mask[:,uidx,0]))
        test_xdata_cur = np.transpose(np.squeeze(test_xdata_a[0:Tcur,uidx,:]),(1,0))
        test_xdata_r=np.squeeze(util.iAugSTFT(test_xdata_cur,129,1,1))
        # append first frames of reference, otherwise reconstruction has artifacts:
        best_xgen_cur = np.transpose(np.squeeze(best_xgen_complete[0:Tcur-npred,uidx,:]),(1,0))
        best_xgen_cur=np.concatenate([test_xdata_cur[:,0:npred].astype(np.float32),best_xgen_cur],axis=1)
        best_xgen_r=np.squeeze(util.iAugSTFT(best_xgen_cur,129,1,1))
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        util.wavwrite(outputfolder+('/est%d.wav'%uidx),np.float32(8000.0),best_xgen_r)
        util.wavwrite(outputfolder+('/ref%d.wav'%uidx),np.float32(8000.0),test_xdata_r)

if __name__ == "__main__":
   main(sys.argv[1:])

