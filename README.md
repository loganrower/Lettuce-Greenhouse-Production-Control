# Lettuce Greenhouse Production Control
Project for Advanced Machine Learning at Wageningen University 

Authors of the Project: Bart van Laatum; Congcong Sun 

Project Group Members: Eric Wiskandt and Logan Rower

Greenhouse model is derived from (Boersma & van Mourik, 2021). 

**Objective:**
- Utilize Reinforcement Learning Algorithms to control the indoor environmental conditions of a lettuce greenhouse with the goal of maximizing net profit.


Boersma S, van Mourik S. 2021. Nonlinear sample-based MPC in a Greenhouse with Lettuce and uncertain weather forecasts. 40th Benelux Workshop on Systems and Control, 58-59.



**Ways to utilize this project:**
- Deterministic Policy Testing in Simulated Greenhouse Environment
- Training and Testing Reinforcement Learning Algorithm within Simulated Greenhouse Environment
- Optimizing the Greenhouse Environment Model

**Before Proceeding with this project:**

> The following steps assume that you will be running this project on your `local machine` as this project and dependencies has yet to be tested in google colab. Also important to note that this project has only been tested on windows machines using Anaconda, and assumes that the user will utilize the provided conda environment to run the project.


**Initilization of Environment and Packages**
-  This project assumes that the user is well versed in using Anaconda for package management, but if not follow the following link to install the Anaconda package distribution system
(https://www.anaconda.com/).

- After installation you are ready to proceed to creating your project environment. See the `condaenv` markdown file for detailed instructions regarding this procedure. When complete with this procedure continue with the steps below for how to utilize the rest of this project, and is available.


**Method 1: Deterministic Policy Testing**
- Files:
    * `run_env_disc.ipynb`
    * `env_disc.py`
- A separate gymnasium environment was created for this deterministic policy. To examine the policy that is used go into the `env_disc.py` file. To see the performance of this deterministic policy run the Jupyter Notebook `run_env_disc.ipynb`. This Jupyter Notebook will provide some statistics and plots for the user.


**Method 2: Training PPO Reinforcement Learning Algorithm**
- Files:
    * environment.py
    * gh_learn.py

- To train any Reinforcment Learning Algorithm in this project first go to the `environment.py` file which illustrates the simulation time in days, start day as well as other parameters that have already been preset. These parameters can be changed within the `LettuceGreenhouse` class for whatever you would like to test. 

- Within the `environment.py` file change the reward function, this is as simple as changing the returned object to be either `reward` or `net_profit`. See the report for more details on the equations of these two reward functions.

- Once the parameters of the environment and the reward function has been initialized you can then proceed to running the `gh_learn.py` script. This script will run for 50000 timesteps and will continuously check for the best model every 1000 steps. In the end a singluar model will have been saved. which can be viewed in the models directory. There are already some other presaved models in that directory and if you would like to view that model or others please use the following terminal command after entering you directory to view tensorboard on the web. 

>                   tensorboard --logdir ./logs/

- This training is so far only been conducted with PPO

**Method 3: Evaluating the trained models**
- Files:
    * `eval_env.py`
    * `gh_eval.py`
- First ensure that the evalulation environment has the parameters appropriately set for the experiment you would like to run. Next, go into the `gh_eval` file. This file will save plots into the `Best_Plots` directory, and specifically `eval_model`. There are currently some plots from previous runs that have been documented in `bestmodels.md` so if wanting to run one of those models you should achieve the same results if you change the zip file that you are loading in to the appropriate name from `bestmodels.md` or if there is a newer model that you would like to run then go into the models directory and copy the relative path for the zip file. 


**Report**

 If wanting to view the report on this project then see the report in the report directory. This report outlines various experiments that were conducted with these environments, and analyzes the various reward functions and methods performed. 

