## REFERENCE FILE...
import optuna
from stable_baselines3 import PPO
from environment import LettuceGreenhouse
from stable_baselines3.common.evaluation import evaluate_policy
import time
import os
# create the environment object for the greenhouse environment
gh =  LettuceGreenhouse()
# initialize the greenhouse environment
gh.reset()

def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float('ent_coef', 0.0, 0.2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    # Create and train the PPO model with the current hyperparameter configuration
    model = PPO("MlpPolicy", gh, learning_rate=learning_rate, ent_coef=ent_coef, batch_size=batch_size)
    model.learn(total_timesteps=10000)  # Adjust the total_timesteps as needed

    # Evaluate the model's performance on the validation set
    validation_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)  # Replace with your validation evaluation function

    # Return the evaluation metric value to Optuna (in this case, cumulative net profit)
    return validation_reward

# Create the Optuna study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)  # Adjust the number of trials as needed

# Retrieve the best hyperparameter configuration
best_params = study.best_params
best_learning_rate = best_params['learning_rate']
best_ent_coef = best_params['ent_coef']
best_batch_size = best_params['batch_size']


models_dir = f"models_method2/{int(time.time())}/"
# Directory for the logs
logdir = f"logs_method2/{int(time.time())}/"

# Create the Directories if they don't already exist
if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

# Train the final model with the best hyperparameters
final_model = PPO("MlpPolicy", gh, learning_rate=best_learning_rate, ent_coef=best_ent_coef, batch_size=best_batch_size, tensorboard_log=logdir)
final_model.learn(total_timesteps=200000)  # Adjust the total_timesteps as needed
final_model.save(f"{models_dir}/{int(time.time())}")