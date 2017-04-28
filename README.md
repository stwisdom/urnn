# urnn
Code for paper "Full-Capacity Unitary Recurrent Neural Networks." Based on the complex_RNN repository from github.com/amarshah/complex_RNN.

Code coming soon for other experiments.

If you find this code useful, please cite the following references:

[1] M. Arjovsky, A. Shah, and Y. Bengio, “Unitary Evolution Recurrent Neural Networks,” Proc. International Conference on Machine Learning (ICML), 2016, pp. 1120–1128.

[2] S. Wisdom, T. Powers, J.R. Hershey, J. Le Roux, and L. Atlas, "Full-Capacity Unitary Recurrent Neural Networks," Advances in Neural Information Processing Systems (NIPS), 2016.

## Instructions for TIMIT prediction experiment

1) Downsample the TIMIT dataset to 8ksamples/sec using Matlab by running ```downsample_audio.m``` from the ```matlab``` directory. Make sure you modify the paths in ```downsample_audio.m``` for your system.

2) Download Matlab evaluation code using ```download_and_unzip_matlab_code.py```, which should download and unzip all the required toolboxes to the ```matlab``` folder.

3) Run the experiments using the shell scripts: ```run_timit_prediction_<model>.sh```, which will train the model and score the resulting audio using the Matlab evaluation toolboxes.
