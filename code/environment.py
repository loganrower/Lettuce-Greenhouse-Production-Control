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
        ## action space
        ###continuous range of actions
        ### upper and lower values are specified as -1 and 1
        ###actions are represented as array of size of control inuts
        self.action_space = spaces.Box(low=-1*np.ones(nu, dtype=np.float32), high=np.ones(nu, dtype=np.float32))
        ## state space
        ### continuous space given with no upper or lower bounds
        ### shape is of ny+nd*Np 
        #### state space will contain a concatenation of ny greenhouse measurement variables 
        #### and nd*Np weather variables.
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
        next_state = self.f(action_denorm, self.d)
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
        reward = self.reward_function(next_state, action_denorm)
        # 5. return observation, reward, done, info
        # return obs, reward, done, {}
        ### dont need to worry about info it can just be an empty dictionary
        obs = next_state
        return obs, reward, done, {}

    def reward_function(self, obs, action):
        """
        This function computes the reward for the greenhouse environment.
        Is called after simulating the environment for one timestep.
        Uses observation and action to compute reward, e.g., profit of greenhouse.
        Args:
            - obs: observation of environment (should be 4 values)
            - action: action of agent
        
        Returns: reward

        Ex: maximize profit of greenhouse
        - need to maximize the lettuce dry weight -> This is a state...
        - need to minmize energy supply by heating system, ventilation rate, and supply rate of CO2 -> these are actions...
        
        total revenue - total expenses
        """

        #TODO: implement reward function.
        # Main goals of this functions are to:
        # 1. Compute reward of greenhouse (e.g., profit of the greenhouse)
        ## first compute the total expense

        ### cost of CO2 (CO2 added)
        ##### cost of CO2 = CO2_Cost [€ kg^{-1}{CO2}] *  Supply Rate of CO2[mg/m2/s]
            ### What about using the CO2 Supply Rate.... This is more with respect to the cost to supply CO2...
            ### What about amount observed indoors as apart of the state? Amount of CO2 Observed Indoors (state[1])[kg/m3]
            ### What about CO2_Capacity [m^3{air} m^{-2}{gh}]
        co2_units = 1/(1000*1000*self.h) # convert action to kg and divide by the amount of time elapsed in the timestep (seconds)
        cost_CO2 = self.p["co2Cost"] * action[0]*(1/(1000*1000*self.h)) # euro/m^2 Cost CO2
        ### COST OF ENERGY:
        """
        Cost of Energy = Cost of Lighting + [Cost of Ventilation] + Cost of Heating
        ## Will likely ommit the cost of lighting for now...

        Ventilation ( NOT INCLUDING FOR NOW)
        - Ventilation Capacity
        - Ventilation Rate
        - Then Cost of Energy

        """
        ## Heating Energy Costs

        # ## What else to energy because that would be an action right not a state? if energy consumed was a state then that would work
                #### but there is no energy state just an action
        
        ### ventilation cost
        
        #### first need to compute the total ventilation rate to understand the intended and unintended airflow in the system
        # ### This means including the leakage and ventilation rate together 
        # ### tot_ventilation = ((Ventilation Rate [mm/s])*(1 m/1000mm)) +  Ventilation Leakage [m/s]
        tot_vent = (action[1]/1000) + self.p["leak"]
        #### Now our final cost equation related to energy expenditure for ventilation is as follows:
        #### The Ventilation Capacity[J m^{-3}°C^{-1}]  * (Total Ventilation Rate (I/O) [m/s])* Cost of Energy [euro/J] *Indoor Air Temp (Current) [°C] * timestep (s)
        cost_vent = self.p["ventCap"]*tot_vent*self.p["energyCost"]*self.state[2]*self.h #euro/m^2 Cost of Energy Related to Ventilation
        
        ### The cost of heating
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [W/m^2] (Convert from W to Joule/s)
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [J/(s*(m^2))] * timestep (seconds)
        heat_cost =  self.p["energyCost"] * action[2] * self.h ## [e]uro/m^2]

        ### Total cost of energy [euro/m^2]
        total_cost_energy = heat_cost + cost_vent

        ## total expenses [euro/m^2]
        total_expenses = total_cost_energy + cost_CO2

        ## next compute thte total revenue
        ### Revenue is determined based on dry weight of crop, yield and price of crop
        ### Yield -> It quantifies the proportion of lettuce produced relative to the dry weight.
        ### Dry Weight -> weight of lettuce after removing the moisture content
        ### combining the yield factor and dry weight gives a better indication of the amount of lettuce to be sold
        ### Price of Lettuce -> euro/kg
        ### Equation for Lettuce Profit = Lettuce Dry Weight [kg/m^2] * Yield Factor [-] * Price of Lettuce [euro/kg]
        ### THE LETTUCE PRICE PARAMETER SEEMS VERY STRANGE.... 
        ### INSTEAD OF LETTUCE PRICE PARAMETER SHOULD I USE self.p["productPrice1"] which is around .81 but the units are also confusing for this ASK BART!
        total_revenue = self.state[0]*self.p["alfaBeta"] *self.p["lettucePrice"]  # [euro/m^2]

        # 2. return reward
        net_profit =  total_revenue - total_expenses


        return net_profit

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
        self.state = np.array([0.0035, 1e-3, 15, 0.008], dtype=np.float32)
        # 2. Reset variables of environment to initial values
        self.timestep = 0
        # 3. Return first observation
        
        
        return self.state

    def close(self):
        return
    ## Adding in the Policy Function HERE...
    def policy_function(self):
        ## compute the 
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
