# Conditional Latent ODEs for Motion Prediction in Autonomous Driving
### This is term project for AI618 course: Generative models and unsupervised learning. 

![Alt text](demo_trajs.gif)

## Installation

#### Simple installation with docker

Please follow [these instructions](https://hub.docker.com/r/kim4375731/clode) to install the environments needed with docker.
After that, you can directly train and test our implementation provided in the docker environment.

#### Manual installation
This code is implemented in Unix system with CUDA 9.0.

In your conda virtual environment, install pytorch 1.1.0 following this

    conda install pytorch==1.1.0 torchvision cudatoolkit=9.0 -c pytorch

After that, using pip to install all dependent packages

    pip install -r requirements.txt

This is all you need for training. However, to evaluate our model by simulation, it is required to install [ngsim environment](https://github.com/sisl/ngsim_env). If you have some trouble with this installation, please contact us!

## Training

All setups for the parameters of training, model, dataset are in file `config.py`.
To train our model with default parameters, simply run:

    python train.py

## Evaluation

#### Multi-agent simulation
We provide a simulation to evaluate our model compared to the baseline AGen. 
The simulation environment is similar to `ngsim_env`. 

    python test.py --test_mode simulation --test_datapath datasets/ngsim_22agents.h5 --ckpt_path pretrained/checkpoint-epoch100.ckpt --use_multi_agents --n_procs 5 --sim_max_obs 20

Our simulation takes around 6-7 hours to finish when only using 1 process.
To speed up time, we recommend to use processes as much as possible depending on your hardware. 
In our cases, we used 5 processes and took around 1.5 hours to finish. 


#### Examples of generated trajectories
To visualize some samples of generated trajectories, follow this command

    python test.py --test_mode visualization --test_datapath datasets/ngsim_22agents.h5 --use_multi_agents --ckpt_path pretrained/checkpoint-epoch100.ckpt --max_obs_length 150 --save_viz_figures
 
    
#### Notes about GPU memory consumption
Simply run `nvidia-smi` while the program is running. The results might vary depending on which kind of NVIDIA card.

The CUDA support for the baseline AGen is not installed in docker environment because of package conflict.

## Acknowlegement
The `models` part of this code is mainly based on [latent-ODE](https://github.com/YuliaRubanova/latent_ode). We specially thank the authors for useful code!
