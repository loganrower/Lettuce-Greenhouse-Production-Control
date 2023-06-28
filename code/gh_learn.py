from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO
import os
from environment import LettuceGreenhouse
import time
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

# Start the timer
start_time = time.time()



#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
        #self.writer.add_scalar('Points', my_points[-1], self.num_timesteps)  # Log the latest point to TensorBoard

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


        # ### get supply co2
        # plt.plot(info['timestep_plot'], info['supply_co2_plot'])
        # plt.title('')
        # plt.xlabel('Time in 15 min steps')
        # plt.ylabel('Supply rate of carbon dioxide [mg]/[m^2][s]')
        # plot_co2_supply = plt.gcf()
        # plt.close()
        # ### get the 
        # plt.plot(info['timestep_plot'], info['indoor_co2_plot'])
        # plt.title('')
        # plt.xlabel('Time in 15 min steps')
        # plt.ylabel('Indoor CO¬2 concentration [ppm]')
        # plot_co2_indoor = plt.gcf()
        # plt.close()

        # # Generate the plot
        # self.writer.add_figure('Supply Rate of CO2', plot_co2_supply, self.num_timesteps)  # Log the plot to TensorBoard
        # self.writer.add_figure('Indoor CO2', plot_co2_indoor, self.num_timesteps)  # Log the plot to TensorBoard

        return True  # Continue training
class SaveOnBestTrainingRewardCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_freq == 0:
            mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                if self.save_path is not None:
                    self.model.save(os.path.join(self.save_path, 'best_model_net_prof_7'))
                    if self.verbose > 0:
                        print(f"Saved the best model with mean reward: {mean_reward:.2f}")
        return True
class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    :param eval_env: (gym.Env) The environment used for initialization
    :param n_eval_episodes: (int) The number of episodes to test the agent
    :param eval_freq: (int) Evaluate the agent every eval_freq call of the callback.
    """

    def __init__(self, eval_env, n_eval_episodes=5, eval_freq=20):
        super().__init__()
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf

    def _on_step(self):
        """
        This method will be called by the model.

        :return: (bool)
        """

        # self.n_calls is automatically updated because
        # we derive from BaseCallback
        if self.n_calls % self.eval_freq == 0:
            # === YOUR CODE HERE ===#
            # Evaluate the agent:
            # you need to do self.n_eval_episodes loop using self.eval_env
            # hint: you can use self.model.predict(obs, deterministic=True)

            # Save the agent if needed
            # and update self.best_mean_reward

            print("Best mean reward: {:.2f}".format(self.best_mean_reward))

            # ====================== #
        return True

# Define hyperparameters
hyperparameters = {
    'learning_rate': 0.001,
    'ent_coef': 0.01,
    'n_steps': 96, # number of steps before policy update... so go a full day...
    'n_epochs': 500, # number of times collected experience will be used for updating policy
    'vf_coef': 0.2,
    'n_epochs': 10,
    'clip_range': 0.2,
    'gamma': 0.95
}
# Create the callback object
callback = CustomCallback()

# Create the callback: check every 1000 steps
## check every 1000 steps to see if ideal model has been found....
callback_best = SaveOnBestTrainingRewardCallback(check_freq=1000, save_path=models_dir)
# create the environment object for the greenhouse environment
gh =  LettuceGreenhouse()


## Start Day range from 0 to 300
# 1 day to 4 weeks which is growing cycle...

gh = LettuceGreenhouse()
gh.reset()
# env = Monitor(gh,logdir)
# Define the Model
### it is a PPO model
### the policy is set to multi perceptron -> keep this consistent for most cases...
modelppo = PPO('MlpPolicy', gh,**hyperparameters,verbose=1, tensorboard_log=logdir)

# # Timesteps
TIMESTEPS = 50000
# for i in range(5):
# Pass the callback object to the model's `learn()` method
modelppo.learn(total_timesteps=TIMESTEPS, callback = callback_best)
    # now save model

# modelppo.save(f"{models_dir}/{TIMESTEPS*i}")

#modelppo= PPO.load("models/1687452562/10000.zip",env = gh )
# mean_reward, std_reward = evaluate_policy(modelsave, modelsave.get_env(), n_eval_episodes=10)
# print(f"The mean reward is {mean_reward} and the standard deviation of the reward is {std_reward}")

# modelppo.learn(total_timesteps=TIMESTEPS, callback=EvalCallback(eval_env = gh),
#             tb_log_name="PPO Eric 2")

# Stop the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print("Elapsed time: {:.2f} seconds".format(elapsed_time))