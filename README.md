# Conditional Latent ODEs for Motion Prediction in Autonomous Driving
### This is term project for AI618 course: Generative models and unsupervised learning. 

![Alt text](demo_trajs.gif)

## Requirements

In your conda virtual environment, install pytorch 1.1.0 following this

    conda install pytorch==1.1.0 torchvision cudatoolkit=9.0 -c pytorch

After that, using pip to install all dependent packages

    pip install -r requirements.txt

## Training

All setups for the parameters of training, model, dataset are in file `config.py`.
To train our model with default parameters, simply run:

    python train.py

## Evaluation

We provide a simulation to evaluate our model compared to the baseline AGen. 
The simulation environment is similar to `ngsim_env`. 

    python test.py --test_mode simulation --test_file_path --ckpt_path pretrained/checkpoint-epoch100.ckpt --use_multi_agents --n_procs 4

Our simulation takes around 6-7 hours to finish when only using 1 process.
To speed up time, we recommend to use processes as much as possible depending on your hardware. 
In our cases, we used 5 processes and took around 1.5 hours to finish. 
