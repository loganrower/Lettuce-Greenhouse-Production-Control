from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
import os
from environment import LettuceGreenhouse
import time
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# The models directory
models_dir = f"models/{int(time.time())}/"
# Directory for the logs
logdir = f"logs/{int(time.time())}/"

# Create the Directories if they don't already exist
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)
	

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.writer = SummaryWriter(log_dir=logdir)  # Create a SummaryWriter object for TensorBoard logging

    def _on_step(self):
        # Update and record your points or generate the plot here
        # Let's say you have a list of points stored in `my_points`
        my_points = [1, 2, 3, 4, 5]
        self.writer.add_scalar('Points', my_points[-1], self.num_timesteps)  # Log the latest point to TensorBoard

        # Saving Figure to Tensorboard...
        ## TEST...
        # a = np.random.randint(1,100,10)
        # b = np.arange(0,10)
        # plt.plot(b,a)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.title('My Plot')
        # plot_object = plt.gcf()
        # # Generate the plot
        # self.writer.add_figure('Plot', plot_object, self.num_timesteps)  # Log the plot to TensorBoard

        # Get data from info
        ### check to see name of info directory......
        ##print(self.locals)
        info = self.locals['infos'][0]
        #print(info)
        ## Now index to get the values we want...
        ### get timestep


        ### get supply co2
        #print(info['timestep_plot'])
        plt.plot(info['timestep_plot'], info['supply_co2_plot'])
        plt.title('')
        plt.xlabel('Time in 15 min steps')
        plt.ylabel('Supply rate of carbon dioxide [mg]/[m^2][s]')
        plot_co2_supply = plt.gcf()
        plt.close()
        plt.plot(info['timestep_plot'], info['indoor_co2_plot'])
        plt.title('')
        plt.xlabel('Time in 15 min steps')
        plt.ylabel('Indoor CO¬2 concentration [ppm]')
        plot_co2_indoor = plt.gcf()
        # # Generate the plot
        self.writer.add_figure('Supply Rate of CO2', plot_co2_supply, self.num_timesteps)  # Log the plot to TensorBoard
        self.writer.add_figure('Indoor CO2', plot_co2_indoor, self.num_timesteps)  # Log the plot to TensorBoard



        return True  # Continue training

# Create the callback object
callback = CustomCallback()

# create the environment object for the greenhouse environment
gh =  LettuceGreenhouse()
# initialize the greenhouse environment
gh.reset()

# Define the Model
### it is a PPO model
### the policy is set to multi perceptron -> keep this consistent for most cases...
model = PPO('MlpPolicy', gh, verbose=1, tensorboard_log=logdir)

# Timesteps
TIMESTEPS = 10000
for i in range(1,2):
    # Pass the callback object to the model's `learn()` method
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False ,callback=callback)

    # now save model
    model.save(f"{models_dir}/{TIMESTEPS}")


