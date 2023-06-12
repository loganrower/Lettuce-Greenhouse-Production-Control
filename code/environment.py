"""
This file provides the environment for a lettuce greenhouse production system.
The AML-project on greenhouse control with RL can use this as a starting point.

Greenhouse dyanmics are modelled as a class of the OpenAI Gym environment.
The controller for this environment control the valves of the greenhouse.
That regulates amount of heating (W/m2) and carbon dioxide into the greenhouse.
""" 
import numpy as np
from utils import co2dens2ppm, vaporDens2rh, load_disturbances, DefineParameters

import gym
from gym import spaces

class LettuceGreenhouse(gym.Env):

    def __init__(self, 
        weather_data_dir="weatherData\outdoorWeatherWurGlas2014.mat",
        ny=4,                 # number of greenhouse measurement variables
        nx=4,                 # number of state variables
        nd=4,                 # number of disturbance (weather variables)
        nu=3,                 # number of control inputs
        h=15*60,              # sampling period (15 minutes, 900 seconds...)
        c=86400,              # conversion to seconds
        nDays=2,              # simulation days
        Np=20,                # number of future predictions (20 == 5hrs)
        startDay=40,          # start day of simulation
        ):
        """
        Greenhouse environment class, implemented as an OpenAI gym environment.

        Args:
            weather_data_dir -- filename of weather data
            h                -- finite differencing step
            c                -- number of seconds in a day
            nDays            -- number of simulation days
        """
        # make inheritance class from gym environment
        super(LettuceGreenhouse, self).__init__()

        # simulation parameters
        self.h = h
        self.c = c
        self.nDays = nDays
        self.L = nDays*c # seconds...
        self.N = self.L//self.h ## 192... steps?

        # action and observation spaces
        self.action_space = spaces.Box(low=-1*np.ones(nu, dtype=np.float32), high=np.ones(nu, dtype=np.float32))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ny + nd*Np,))

        # lower and upper bounds on observations
        self.obs_low = np.array([0., 0., 10., 0.], dtype=np.float32)
        self.obs_high = np.array([7., 1.6, 20., 70.], dtype=np.float32)

        # lower and upper bounds on the actions
        self.min_action = np.array([0., 0., 0.], dtype=np.float32)
        self.max_action = np.array([1.2, 7.5, 150.], dtype=np.float32)

        self.p = DefineParameters()

        # initial state of the environment
        self.state = np.array([0.0035, 1e-3, 15, 0.008], dtype=np.float32)
        self.timestep = 0

        # number of variables
        self.Np = Np
        self.ny = ny
        self.nx = nx
        self.nd = nd

        # loadin weather predictions
        self.d = load_disturbances(c, self.L, h , nd, Np, startDay, weather_data_dir)

    def step(self, action):
        """
        Step function that simulates one timestep into the future given input action.

        Args:
            actions [array] -- normalised action

        Return:
            observation [array] -- array consisting of four variables
            reward      [float] -- immediate reward of the environment
            done        [bool]  -- whether state is terminal
            info        [dict]  -- additional information
        """
        #TODO: implement step function.
        # Main goals of this functions are to:
        # 1. Denormalise action
        ## The actions were converted into a space where they have been normalized
        ## Normalization -> x_norm = (x - xmin) / (xmax - xmin)
        ## Denormalization -> x = x_norm*(xmax - xmin) + xmin
        ## So the normalization was done into make it easier for agent to explore the action space
        ## Need to convert back so then we can consider the environment 

        action_denorm = action*(self.max_action - self.min_action) + self.min_action

        # 2. Transition state to next state given action and observe environment
        ## Observe the environment or the weather?
        state_new = f(action_denorm, self.d)
        # 3. Check whether state is terminal
        ## how do we know if it is a terminal state... based on if end of simulation so if it has been 2days...
        ## so we will just add one to the timestep since there are 192 periods that we are sampling from
        ## 
        if self.N == self.timestep:
            done =  "done"
        else:
            done = "not done"

        # 4. Compute reward from profit of greenhouse
        ## how good action was....
        ## can determine where we want to focus...
        ### Ex: focus on minimizing heating and environmental cost...
        ### Ex: focus on the production of lettuce...
        ### The function will then steer what the algorithm will focus on...
        reward = self.reward_function(obs, action_denorm)
        # 5. return observation, reward, done, info
        # return obs, reward, done, {}
        ### dont need to worry about info it can just be an empty dictionary
        return obs, reward, done, {}

    def reward_function(self, obs, action):
        """
        This function computes the reward for the greenhouse environment.
        Is called after simulating the environment for one timestep.
        Uses observation and action to compute reward, e.g., profit of greenhouse.
        Args:
            - obs: observation of environment
            - action: action of agent
        
        Returns: reward

        Ex: profit of greenhouse

        Ex: production of the lettuce..


        """
        #TODO: implement reward function.
        # Main goals of this functions are to:
        # 1. Compute reward of greenhouse (e.g., profit of the greenhouse)
        # 2. return reward
        return 

    def reset(self):
        """
        Resets environment to starting state.
        Returns:
            observation -- environment state

        called every time before you run your environment

        and reset variables to their initial state...

        
        """
        #TODO: implement reset function.
        # Main goals of this functions are to:
        # 1. Reset state of environment to initial state
        # 2. Reset variables of environment to initial values
        # 3. Return first observation
        pass

    def close(self):
        return

    def f(self, action, d):
        """
        State transition function.
        Args:
            x  --   state variables
            u  --   control variables
            d  --   (weather) disturbances
            p  --   parameters(?)
            h  --   finite difference step

        Returns:
            x   --  new state variables
        """
        # finite differencing method to compute new state variables
        k1 = self.F(self.state, action, d, self.p)
        k2 = self.F(self.state+self.h/2 *k1, action, d, self.p)
        k3 = self.F(self.state+self.h/2 *k2, action, d, self.p)
        k4 = self.F(self.state+self.h *k3, action, d, self.p)
        self.state += self.h/6*(k1+ 2*k2 + 2*k3 + k4)
        return self.state

    def g(self):
        """
        Function that provides the measurements of the environment.
        Give a more readable output of the state variables.

        Returns
            y   --  measurements of the environment
        """ 
        y = np.array([1e3*self.state[0],
                1e-3*co2dens2ppm(self.state[2],self.state[1]),
                self.state[2],
                vaporDens2rh(self.state[2], self.state[3])], dtype=np.float32)
        return y

    def F(self, x, u, d, p):
        """
        Function to compute change change of x variables
        Args:
            x   --   state variables
            u   --   control variables
            d   --   (weather) disturbances
            p   --   parameters(?)

        returns:
            delta x --   change of state variables
        """
        # way to compute next time step
        ki =  np.array([
            p["alfaBeta"]*(
            (1-np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
            (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"]) 
            / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])))
            - p["Wc_a"] * x[0] * 2**(0.1 * x[2] - 2.5)
            ,

            1 / p["CO2cap"] * (
            -((1 - np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
            (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])
            / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2]**2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])))
            + p["CO2c_a"] * x[0] * 2**(0.1 * x[2] - 2.5) + u[0]/1e6 - (u[1] / 1e3 + p["leak"]) * (x[1] - d[1])
            ),

            1/p["aCap"] * (
            u[2] - (p["ventCap"] * u[1] / 1e3 + p["trans_g_o"]) * (x[2] - d[2]) + p["rad_o_g"] * d[0]
            ),

            1/p["H2Ocap"] * ((1 - np.exp(-p["laiW"] * x[0])) * p["evap_c_a"] * (p["satH2O1"]/(p["R"]*(x[2]+p["T"]))*
            np.exp(p["satH2O2"] * x[2] / (x[2] + p["satH2O3"])) - x[3]) - (u[1]/1e3 + p["leak"]) * (x[3] - d[3]))]
            )
        return ki
