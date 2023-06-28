from stable_baselines3.common.env_checker import check_env
from environment import LettuceGreenhouse


# Custom Environment in our case it is LettuceGreenhouse, no parameters are passed...
env = LettuceGreenhouse()
# It will check your custom environment and output additional warnings if needed
check_env(env)