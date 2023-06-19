from stable_baselines3 import PPO
import os
from environment import LettuceGreenhouse
import time

# The models directory
models_dir = f"models/{int(time.time())}/"
# Directory for the logs
logdir = f"logs/{int(time.time())}/"

# Create the Directories if they don't already exist
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
	
# create the environment object for the greenhouse environment
gh =  LettuceGreenhouse()
# initialize the greenhouse environment
gh.reset()

# Define the Model
### it is a PPO model
### the policy is set to multi perceptron -> keep this consistent for most cases...
model = PPO('MlpPolicy', gh, verbose=1, tensorboard_log=logdir)

# Training the Model

## Set the Timesteps
TIMESTEPS = 10000
### how many total timesteps...10000 timesteps for training

# How many timesteps do we want to do... so we are going to run this for a really long time to see our results..
## in our case we could do a while True, or even just a for loop that would do many iterations of 10000 so in this case I will do 100 so up to 1,000,000 timesteps essentiallt
for i in range(1,10):

    # model will reset the number of timesteps that it has already taken the model object
    # model.learn would reset the timesteps it is at but... do reset_num_timesteps = FALSE
    # tb_log_name = "PPO" <- descriptor
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name = "PPO") # training every 10000 timesteps...

    # now save model
    model.save(f"{models_dir}/{TIMESTEPS*i}")
