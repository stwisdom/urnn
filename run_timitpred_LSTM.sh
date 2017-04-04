#!/bin/bash

gpuIdx=0

dataset="timit_trainNoSA_dev_coreTest"
prng_seed_Givens=51026

indir="TIMIT_8khz"

niter=13000 #number of training mini-batch iterations
nAllowedIncs=100 #number of increases in validation loss allowed (in units of epochs)
nbatch=28 #30 yields 154 iterations per epoch
nhidden=65
lr=0.001
lr_print="${lr/./-}"
model="LSTM"
datatype="complex"
cost="MSE"
fold="fold1"
scene="all"
flag_feed_forward=0
flag_generator=0
downsample_train=1
downsample_test=1
for num_pred_steps in 1; do

    echo "npred=${num_pred_steps}"
    
    for nhidden in 84 120 156; do
        #for seed in 1234 2345 3456; do
        for seed in 1234; do
            for flag_useFullW in 0; do

                if (( flag_useFullW == 0 )); then
                    Wimpl="adhoc"
                else
                    Wimpl="full"
                fi
 
                savefile="exp/timit_prediction_trainNoSA-dev-coreTest_ae_niter${niter}_nbatch${nbatch}_nhidden${nhidden}_lr${lr_print}_${model}_${datatype}_${cost}_${fold}_${scene}_flagff${flag_feed_forward}_flaggen${flag_generator}_dstrain${downsample_train}_dstest${downsample_test}_${Wimpl}_prngSeed${prng_seed_Givens}_nAllowedInc${nAllowedIncs}_itsPerValid130_hbias0-0_logmag_npred${num_pred_steps}_seed${seed}"
 
                echo "Running experiment, writing to savefile ${savefile}"
                echo "===============" 
                cmd="THEANO_FLAGS='device=gpu${gpuIdx}' python -u timit_prediction.py ${niter} ${nbatch} ${nhidden} ${lr} ${savefile} ${model} ${datatype} True ${cost} ${fold} ${scene} --indir ${indir} --dataset ${dataset} --flag_feed_forward ${flag_feed_forward} --flag_generator ${flag_generator} --downsample_train ${downsample_train} --downsample_test ${downsample_test} --prng_seed_Givens ${prng_seed_Givens} --num_allowed_test_inc ${nAllowedIncs} --iters_per_validCheck 130 --flag_useFullW ${flag_useFullW} --num_pred_steps ${num_pred_steps} --hidden_bias_mean 0.0 --data_transform logmag --offset_eval 400 --n_utt_eval_spec 192 --seed ${seed}"
            
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

