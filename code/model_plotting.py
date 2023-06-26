
from stable_baselines3 import PPO
from environment import LettuceGreenhouse
import os
# import time
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
# import numpy as np

seed_value = 42  # Replace with your desired seed value
set_random_seed(seed_value)

# Loaded in the model that was desired
## This model version is the model that has +1, -1 reward and +100 reward at specific state
#model = PPO.load("models_method2/1687571802/1687572264.zip")
model = PPO.load("models/1687696845/best_model.zip")
# create the environment
gh = LettuceGreenhouse()

# intitialize the environment and get the initial obsevation
obs = gh.reset()

# next we will get the timesteps for a single episode:
time_steps = gh.N

# now we will get the data that we want to collect in form of lists
## States (obs)
dry_weight =[]
indoor_co2 = []
indoor_temp= []
rh = []
## Actions (actions)
supply_co2 = []
vent = []
supply_energy = []
## episode Rewards
ep_rewards = [] # cumulative per episode
# total time passed in episode..
timestep = []

# Function to save the plot for a single episode
def ep_reward(timestep,ep_rewards):
    save_path = f'Best_Plots/final_model/'
    os.makedirs(save_path, exist_ok=True) 
    plt.figure(figsize=(15,15))
    plt.plot(timestep,ep_rewards)
    #ax_1.set_title('')
    save_path = f'Best_Plots/final_model/'
    plt.xlabel('Time in 15 min steps')
    plt.ylabel('Cumulative Reward over 1 Episode')
    plt.savefig(save_path + f'reward_plot.png')
    plt.close()  # Close the plot window after each plot

def ep_plots(timestep,dry_weight,indoor_co2,indoor_temp,rh,supply_co2,vent,supply_energy):
    print("Plotting")
    ## Make a new directory for episode
    save_path = f'Best_Plots/final_model/'
    os.makedirs(save_path, exist_ok=True) 
    # Plotting code for this episode's data
    # Customize the plot as needed
    plt.figure(figsize=(30,30))

    ax_1 = plt.subplot(4,2,1)
    plt.plot(timestep, supply_co2)
    #ax_1.set_title('')
    ax_1.set_xlabel('Time in 15 min steps')
    ax_1.set_ylabel('Supply rate of carbon dioxide [mg]/[m^2][s]')

    ax_2 = plt.subplot(4,2,2)
    plt.plot(timestep, vent)
    ax_2.set_xlabel('Time in 15 min steps')
    ax_2.set_ylabel('Ventilation rate [mm]/[s]')

    ax_3 = plt.subplot(4,2,3)
    plt.plot(timestep,supply_energy)
    ax_3.set_xlabel('Time in 15 min steps')
    ax_3.set_ylabel('Energy supply by heating the system [W]/[m^2]')

    ax_4 = plt.subplot(4,2,4)
    plt.plot(timestep,dry_weight)
    ax_4.set_xlabel('Time in 15 min steps')
    ax_4.set_ylabel('Lettuce dry weight [g]/[m^2]')

    ax_5 = plt.subplot(4,2, 5)
    plt.plot(timestep,indoor_co2)
    ax_5.set_xlabel('Time in 15 min steps')
    ax_5.set_ylabel('Indoor CO¬2 concentration [ppm]')

    ax_6 = plt.subplot(4,2, 6)
    plt.plot(timestep,indoor_temp)
    ax_6.set_xlabel('Time in 15 min steps')
    ax_6.set_ylabel('Indoor air temperature [C]')

    ax_7 = plt.subplot(4,2, 7)
    plt.plot(timestep,rh)
    ax_7.set_xlabel('Time in 15 min steps')
    ax_7.set_ylabel('Indoor relative humidity [%]')

    # ax_8 = plt.subplot(4,2, 8)
    # plt.plot(timestep,ep_rew)
    # ax_8.set_xlabel('Time in 15 min steps')
    # ax_8.set_ylabel('Episode Rewards Over Time')

    # Save the figure:
    plt.savefig(save_path + f'action_state_final_plot.png')
    plt.close()  # Close the plot window after each plot

ep_num = 10
done =  False
# for i in range(ep_num):
netprof_val = 0
cum_rew = 0
while not done:
    # first predict the next action using the observation
    action, _ = model.predict(obs)
    # Then take a step using the action and get the new observations, reward, if the episode is done, and the info metrics..
    obs, reward, done, infos = gh.step(action)
    # finally append values to the lists...
    print(done)
    cum_rew += reward
    ep_rewards.append(cum_rew)
    
info =  infos
net_prof = info['net_profit']
##
timestep = info["timestep_plot"] 
dry_weight = info['dry_weight'] 
indoor_co2 = info['indoor_co2'] 
temp = info['temp'] 
rh = info['rh'] 
supply_co2 = info["supply_co2"] 
vent = info['vent_plot'] 
supply_energy = info['supply_energy']

ep_plots(timestep, dry_weight,indoor_co2,temp ,rh,supply_co2,vent,supply_energy)
ep_reward(timestep,ep_rewards)

mean_reward, std_reward = evaluate_policy(model, gh, n_eval_episodes=10)
print(f"The mean reward is {mean_reward} and the standard deviation of the reward is {std_reward}")
print("Net Profit:", net_prof)
print("Cumulative Reward:", cum_rew)