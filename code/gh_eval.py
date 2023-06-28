from stable_baselines3 import PPO
from eval_env import LettuceGreenhouse
import os
# import time
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import numpy as np
from stable_baselines3.common.monitor import Monitor
import copy
import time
# import numpy as np
# Start the timer
start_time = time.time()
seed_value = 120 # Replace with your desired seed value
set_random_seed(seed_value)

# Loaded in the model that was desired
## This model version is the model that has +1, -1 reward and +100 reward at specific state
#model = PPO.load("models_method2/1687571802/1687572264.zip")
model = PPO.load("models/1687872671/best_model_40.zip")
# create the evaluation environment ## Start Day range from 0 to 300
## only initialize random day once.... will then train over this period 10 times...
eval_gh = LettuceGreenhouse()
#eval_gh = Monitor(eval_gh)

# intitialize the environment and get the initial obsevation
obs = eval_gh.reset()

# next we will get the timesteps for a single episode:
time_steps = eval_gh.N +1

# now we will get an array of lists based on number of timesteps in this...
array_zeros = np.empty(time_steps, dtype=object)
for i in range(time_steps):
    # for every index we will change from zero to list...
    array_zeros[i] = []


# now we will get the data that we want to collect in form of lists
## States (obs)


dry_weight = copy.deepcopy(array_zeros)
indoor_co2 = copy.deepcopy(array_zeros)
indoor_temp= copy.deepcopy(array_zeros)
rh = copy.deepcopy(array_zeros)
## Actions (actions)
supply_co2 = copy.deepcopy(array_zeros)
vent =copy.deepcopy(array_zeros)
supply_energy = copy.deepcopy(array_zeros)
## episode Rewards
ep_rewards = copy.deepcopy(array_zeros)# cumulative per episode
# total time passed in episode..
timestep = copy.deepcopy(array_zeros)
net_prof = []

# Function to save the plot for a single episode
def ep_reward(timestep,ep_rewards):
    save_path = f'Best_Plots/eval_model/'
    os.makedirs(save_path, exist_ok=True) 
    plt.figure(figsize=(15,15))
    plt.plot(timestep,ep_rewards)
    #ax_1.set_title('')
    plt.xlabel('Time in 15 min steps')
    plt.ylabel('Cumulative Reward over 1 Episode')
    plt.savefig(save_path + f'reward_plot.png')
    plt.close()  # Close the plot window after each plot

def ep_plots(timestep,dry_weight,indoor_co2,indoor_temp,rh,supply_co2,vent,supply_energy):
    print("Plotting")
    ## Make a new directory for episode
    save_path = f'Best_Plots/eval_model/'
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
    plt.savefig(save_path + f'action_state_eval_plot_1.png')
    plt.close()  # Close the plot window after each plot

ep_num = 100
done =  False


cum_rew = 0
for i in range(ep_num):
    obs = eval_gh.reset()
    print(eval_gh.startDay)
    while not done:
        # first predict the next action using the observation
        action, _ = model.predict(obs)
        # Then take a step using the action and get the new observations, reward, if the episode is done, and the info metrics..
        obs, reward, done, infos = eval_gh.step(action)
        # finally append values to the lists...
        cum_rew += reward
        ep_rewards[eval_gh.timestep-1].append(cum_rew)
        timestep = infos["timestep_plot"]
    # now we will add to the list... of values
    for t in timestep : # number of timesteps...
        dry_weight[t].append(infos['dry_weight'][t])
        
        indoor_co2[t].append(infos['indoor_co2'][t]) 
        indoor_temp[t].append(infos['temp'][t]) 
        rh[t].append(infos['rh'][t]) 
        supply_co2[t].append(infos["supply_co2"][t])
        vent[t].append(infos['vent_plot'][t])
        supply_energy[t].append(infos['supply_energy'][t])
    net_prof.append(infos['net_profit'])
## now compute the averages...
dry_weight = np.array([np.mean(sublist) for sublist in dry_weight])

# print(dry_weight)
indoor_co2 = np.array([np.mean(sublist) for sublist in indoor_co2])
indoor_temp= np.array([np.mean(sublist) for sublist in indoor_temp])
rh = np.array([np.mean(sublist) for sublist in rh])
supply_co2 = np.array([np.mean(sublist) for sublist in supply_co2])
vent= np.array([np.mean(sublist) for sublist in vent])
supply_energy = np.array([np.mean(sublist) for sublist in supply_energy])
ep_rewards = np.array([np.mean(sublist) for sublist in ep_rewards])

print(np.array(timestep).shape)
print(indoor_co2.shape)
# print(ep_rewards)
# # mean net profit
net_prof_mean = np.mean(net_prof) 

print("Mean Net Profit:", net_prof_mean)
print("Cumulative Reward:", cum_rew)
# # info =  infos
# # net_prof = info['net_profit']
# # ##
# # timestep = info["timestep_plot"] 
# # dry_weight = info['dry_weight'] 
# # indoor_co2 = info['indoor_co2'] 
# # temp = info['temp'] 
# # rh = info['rh'] 
# # supply_co2 = info["supply_co2"] 
# # vent = info['vent_plot'] 
# # supply_energy = info['supply_energy']

ep_plots(timestep, dry_weight,indoor_co2,indoor_temp ,rh,supply_co2,vent,supply_energy)
ep_reward(timestep,ep_rewards)

# mean_reward, std_reward = evaluate_policy(model, eval_gh, n_eval_episodes=10)
# print(f"The mean reward is {mean_reward} and the standard deviation of the reward is {std_reward}")
# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))