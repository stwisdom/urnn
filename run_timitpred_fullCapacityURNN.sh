#!/bin/bash

gpuIdx=0

dataset="timit_trainNoSA_dev_coreTest"
prng_seed_Givens=51026

indir="TIMIT_8khz"

# for TIMIT, training set size is 4620 utterances
# factors of 4620: 1  |  2  |  3  |  4  |  5  |  6  |  7  |  10  |  11  |  12  |  14  |  15  |  20  |  21  |  22  |  28  |  30  |  33  |  35  |  42  |  44  |  55  |  60  |  66  |  70  |  77  |  84  |  105  |  110  |  132  |  140  |  154  |  165  |  210  |  220  |  231  |  308  |  330  |  385  |  420  |  462  |  660  |  770  |  924  |  1155  |  1540  |  2310  |  4620   (48 divisors)

# for TIMIT, training set size without SA is 3640 utterances
# factors of 3640:
# 1,2,4,5,7,8,10,13,14,20,26,28,35,40,52,56,65,70,91,104,130,140,182,260

niter=13000 #number of training mini-batch iterations
nepochs=100 #number of increases in validation loss allowed (in units of epochs)
nbatch=28 #28 yields 130 iterations per epoch
nhidden=65
lr=0.001
lr_print="${lr/./-}"
model="complex_RNN"
datatype="complex"
cost="MSE"
fold="fold1"
scene="all"
flag_feed_forward=0
flag_generator=0
downsample_train=1
downsample_test=1
for num_pred_steps in 1; do

    echo "npred=${npred}"
    
    for nhidden in 128 192 256; do
        #for seed in 1234 2345 3456; do
        for seed in 1234; do
            for flag_useFullW in 1; do

                if (( flag_useFullW == 0 )); then
                    Wimpl="adhoc_fast"
                else
                    Wimpl="full"
                fi
            
                savefile="exp/timit_prediction_trainNoSA-dev-coreTest_ae_niter${niter}_nbatch${nbatch}_nhidden${nhidden}_lr${lr_print}_${model}_${datatype}_${cost}_${fold}_${scene}_flagff${flag_feed_forward}_flaggen${flag_generator}_dstrain${downsample_train}_dstest${downsample_test}_${Wimpl}_prngSeed${prng_seed_Givens}_nAllowedInc${nepochs}_itsPerValid130_hbias0-0_logmag_npred${num_pred_steps}_seed${seed}"
            
                echo "Running experiment, writing to savefile ${savefile}"
                echo "===============" 
                cmd="THEANO_FLAGS='device=gpu${gpuIdx}' python2.7 -u timit_prediction.py ${niter} ${nbatch} ${nhidden} ${lr} ${savefile} ${model} ${datatype} True ${cost} ${fold} ${scene} --indir ${indir} --dataset ${dataset} --flag_feed_forward ${flag_feed_forward} --flag_generator ${flag_generator} --downsample_train ${downsample_train} --downsample_test ${downsample_test} --prng_seed_Givens ${prng_seed_Givens} --num_allowed_test_inc ${nepochs} --iters_per_validCheck 130 --flag_useFullW ${flag_useFullW} --num_pred_steps ${num_pred_steps} --hidden_bias_mean 0.0 --data_transform logmag --offset_eval 400 --n_utt_eval_spec 192 --seed ${seed}"

                echo $cmd
                SECONDS=0
                eval $cmd
                echo "Experiment took $SECONDS seconds."
                echo ""

                # score the results
                echo "${savefile}_eval" > exp_list
                cmd="matlab -nosplash -nodesktop -nodisplay -r \"addpath('matlab'); try score_timitpred('exp_list'); catch; end; quit\""
                echo $cmd
                echo "Scoring results of experiment..."
                eval $cmd

            done
        done
    done
done

