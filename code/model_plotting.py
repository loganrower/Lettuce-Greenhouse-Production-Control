
from stable_baselines3 import PPO
from environment import LettuceGreenhouse
import os
# import time
import matplotlib.pyplot as plt
# import numpy as np

seed = 2

# Loaded in the model that was desired
## This model version is the model that has +1, -1 reward and +100 reward at specific state
#model = PPO.load("models_method2/1687571802/1687572264.zip")
model = PPO.load("models/1687607344/10000.zip")
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
def ep_plots(timestep, ep_rewards,dry_weight,indoor_co2,indoor_temp,rh,supply_co2,vent,supply_energy,ep_rew, episode_num):
    print("Plotting")
    ## Make a new directory for episode
    save_path = f'Rew_Method_3_Plots/MinMax_{episode_num}/'
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

    ax_8 = plt.subplot(4,2, 8)
    plt.plot(timestep,ep_rew)
    ax_8.set_xlabel('Time in 15 min steps')
    ax_8.set_ylabel('Episode Rewards Over Time')

    # Save the figure:
    plt.savefig(save_path + f'episode_{episode_num}_plot.png')
    plt.close()  # Close the plot window after each plot

ep_num = 10
done =  False
# for i in range(ep_num):
netprof_val = 0
while not done:
    # first predict the next action using the observation
    action, _ = model.predict(obs)
    # Then take a step using the action and get the new observations, reward, if the episode is done, and the info metrics..
    obs, reward, done, infos = gh.step(action)
    # finally append values to the lists...
    print(done)
    ep_rewards.append(reward)
    timestep.append(gh.timestep)
    dry_weight.append(obs[0])
    indoor_co2.append(obs[1])
    indoor_temp.append(obs[2])
    rh.append(obs[3])
    supply_co2.append(action[0])
    vent.append(action[1])
    supply_energy.append(action[2])

    

net_prof = infos['net_profit']
print("Net Profit:", net_prof)
ep_plots(timestep, ep_rewards,dry_weight,indoor_co2,indoor_temp,rh,supply_co2,vent,supply_energy,ep_rewards,2 )


