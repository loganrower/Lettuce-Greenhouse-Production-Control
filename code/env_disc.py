"""
This file provides the environment for a lettuce greenhouse production system.
The AML-project on greenhouse control with RL can use this as a starting point.

Greenhouse dyanmics are modelled as a class of the OpenAI Gym environment.
The controller for this environment control the valves of the greenhouse.
That regulates amount of heating (W/m2) and carbon dioxide into the greenhouse.
"""
import numpy as np
from utils import co2dens2ppm, vaporDens2rh, load_disturbances, DefineParameters
import matplotlib.pyplot as plt

import gym
from gym import spaces


class LettuceGreenhouse(gym.Env):

    def __init__(self,
                 weather_data_dir="weatherData\outdoorWeatherWurGlas2014.mat",
                 ny=4,  # number of greenhouse measurement variables
                 nx=4,  # number of state variables
                 nd=4,  # number of disturbance (weather variables)
                 nu=3,  # number of control inputs
                 h=15 * 60,  # sampling period (15 minutes, 900 seconds...)
                 c=86400,  # conversion to seconds
                 nDays=48,  # simulation days
                 Np=20,  # number of future predictions (20 == 5hrs)
                 startDay=90,  # start day of simulation
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
        self.h = h  # sampling period, the data is taken every 15 minutes
        self.c = c
        self.nDays = nDays
        self.L = nDays * c  # two simulation days in seconds
        self.N = self.L // self.h  ## 192 steps

        # action and observation spaces
        ## action space
        ###continuous range of actions
        ### upper and lower values are specified as -1 and 1
        ### actions are represented as array of size of control inputs
        ### Normalized Action Space
        ##  # -	Supply rate of carbon dioxide [mg/m2/s]
        ##  # -	Ventilation rate [mm/s]
        ##  # -	Energy supply by heating the system [W/m2]

        self.action_space = spaces.Box(low=-1 * np.ones(nu, dtype=np.float32), high=np.ones(nu, dtype=np.float32))

        ## state space
        ### continuous space given with no upper or lower bounds
        ### shape is of ny+nd*Np 
        #### state space will contain a concatenation of ny greenhouse measurement variables 
        #### and nd*Np weather variables.
        #### Initial Four Measurements... (Current of State Variables)
        #### Then we have Future Predictions for the Four State Variables... (Future Prediction of State Variables)
        #### The observation space is then split up 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ny + nd * Np,))

        # lower and upper bounds on observations
        self.obs_low = np.array([0., 0., 0., 0.], dtype=np.float32)
        self.obs_high = np.array([7., 1.6, 30., 70.],
                                 dtype=np.float32)  # changed max temp to be 30 C cause lettuce can grow at a max of around 29 C

        # lower and upper bounds on the actions
        self.min_action = np.array([0., 0., 0.], dtype=np.float32)
        self.max_action = np.array([1.2, 7.5, 150.], dtype=np.float32)

        self.p = DefineParameters()

        # initial state of the environment
        ## -	Lettuce dry weight [kg/m2]
        ## -	Indoor CO¬2 concentration [kg/m3]
        ## -	Indoor air temperature [C]
        ## -	Indoor humidity [kg/m¬3]

        self.state = np.array([0.0035, 1e-3, 15, 0.008], dtype=np.float32)
        self.state_init = np.array([0.0035, 1e-3, 15, 0.008], dtype=np.float32)
        self.timestep = 0
        self.cum_reward = 0
        ## State Old
        self.old_state = np.array([0.0035, 1e-3, 15, 0.008], dtype=np.float32)
        ## the done state -> for when program is finished
        ### initialize to False
        self.done = False

        # plot variables
        self.dry_weight_plot = []
        self.indoor_co2_plot = []
        self.temp_plot = []
        self.rh_plot = []
        self.supply_co2_plot = []
        self.vent_plot = []
        self.supply_energy_plot = []
        self.timestep_plot = []

        # Weight variable
        self.weight_change = 0

        # number of variables
        self.Np = Np
        self.ny = ny
        self.nx = nx
        self.nd = nd

        # loadin weather predictions
        self.d = load_disturbances(c, self.L, h, nd, Np, startDay, weather_data_dir)

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
        ## The actions were converted into a space where they have been normalized
        ## Normalization -> x_norm = (x - xmin) / (xmax - xmin)
        ## Denormalization -> x = x_norm*(xmax - xmin) + xmin
        action_denorm = (1 + action) * (self.max_action - self.min_action) / (2 + self.min_action)

        print("Action:", action_denorm)
        # 2. Transition state to next state given action and observe environment
        print("Old State:", self.old_state)
        obs = self.f(action_denorm, self.d[self.timestep])

        measurement = self.g()
        self.dry_weight_plot.append(measurement[0])
        self.indoor_co2_plot.append(measurement[1])
        self.temp_plot.append(measurement[2])
        self.rh_plot.append(measurement[3])
        self.timestep_plot.append(self.timestep)
        self.supply_co2_plot.append(action_denorm[0])
        self.vent_plot.append(action_denorm[1])
        self.supply_energy_plot.append(action_denorm[2])
        # print("Current temperature:", obs[2])

        # 3. Compute reward from profit of greenhouse
        reward = self.reward_function(obs, action_denorm)

        # 5. Check whether state is terminal
        ## First see if self.done has been set to True 
        if self.done != True:
            # if it hasnt then check if it is true based on terminal state..
            self.done = self.terminal_state()

        ## Here we need to add in the environmental data to the observation....
        ### for loop 20 times..

        self.old_state = obs
        print("--------", )

        return obs, reward, self.done, {}

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

        ONLY TESTING WITH TOTAL REVENUE FOR ASSIGNMENT STEP 3
        """

        # TODO: implement reward function.
        # Main goals of this functions are to:
        # 1. Compute reward of greenhouse (e.g., profit of the greenhouse)
        ## first compute the total expense

        ### cost of CO2 (CO2 added)
        ##### cost of CO2 = CO2_Cost [€ kg^{-1}{CO2}] *  Supply Rate of CO2[mg/m2*s] * s *(1g/1000mg) *(1kg/1000g)
        #    ### What about using the CO2 Supply Rate.... This is more with respect to the cost to supply CO2...
        #    ### What about amount observed indoors as apart of the state? Amount of CO2 Observed Indoors (state[1])[kg/m3]
        #    ### What about CO2_Capacity [m^3{air} m^{-2}{gh}]
        co2_units = 1 / (
                    1000 * 1000)  # convert action to kg and divide by the amount of time elapsed in the timestep (seconds)
        cost_CO2 = self.p["co2Cost"] * action[0] * (co2_units) * self.h  # euro/m^2 Cost CO2
        ### COST OF ENERGY:
        """
        Cost of Energy = Cost of Lighting + [Cost of Ventilation] + Cost of Heating
        ## Will likely ommit the cost of lighting for now...

        """
        ## Heating Energy Costs

        # ## What else to energy because that would be an action right not a state? if energy consumed was a state then that would work
        #### but there is no energy state just an action

        ### ventilation cost

        #### first need to compute the total ventilation rate to understand the intended and unintended airflow in the system
        # ### This means including the leakage and ventilation rate together 
        # ### tot_ventilation = ((Ventilation Rate [mm/s])*(1 m/1000mm)) +  Ventilation Leakage [m/s]
        tot_vent = (action[1] / 1000) + self.p["leak"]
        #### Now our final cost equation related to energy expenditure for ventilation is as follows:
        #### The Ventilation Capacity[J m^{-3}°C^{-1}]  * (Total Ventilation Rate (I/O) [m/s])* Cost of Energy [euro/J] *Indoor Air Temp (Current) [°C] * timestep (s)
        cost_vent = self.p["ventCap"] * tot_vent * self.p["energyCost"] * obs[
            2] * self.h  # euro/m^2 Cost of Energy Related to Ventilation

        ### The cost of heating
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [W/m^2] (Convert from W to Joule/s)
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [J/(s*(m^2))] * timestep (seconds)
        heat_cost = self.p["energyCost"] * action[2] * self.h  ## [e]uro/m^2]

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
        ##### ORIGINAL EQUATION....
        #####total_revenue = obs[0]*self.p["alfaBeta"] *self.p["lettucePrice"]  # [euro/m^2]
        ##### Equation... https://www.sciencedirect.com/science/article/pii/S0967066108001019#bib22
        ##### auc_price = productPrice1 [euro/m^2] + (Lettuce Dry Weight [kg/m^2] * Dry Weight to Wet Weight Ratio * productPrice2 [euro/(kg)])
        #### auc_price = [euro/m^2] + [euro/m^2]
        if self.timestep == 0:
            total_revenue = (abs(self.old_state[0] - obs[0]) * self.p["productPrice2"]) + self.p[
                "productPrice1"]  # Auction Price of Lettuce euro/m^2
        else:
            # dont add the extra..
            total_revenue = (abs(self.old_state[0] - obs[0])) * self.p[
                "productPrice2"]  # Auction Price of Lettuce euro/m^2

        # 2. return reward|
        net_profit = float((total_revenue) - (total_expenses))
        # print("timestep:",self.timestep)
        # print("Total Rev", total_revenue)
        # print("Total Expenses",total_expenses)
        # print("Weight change: " + str((obs[0]-self.old_state[0])*1000))
        self.weight_change += (obs[0] - self.old_state[0]) * 1000
        # print("Net Profit", net_profit)
        # print("Cumulative Reward", self.cum_reward)
        # self.cum_reward += net_profit
        return net_profit

    def reset(self):
        """
        Resets environment to starting state.
        Returns:
            observation -- environment state
        called every time before you run your environment
        and reset variables to their initial state...

        """
        # TODO: implement reset function.
        # Main goals of this functions are to:
        # 1. Reset state of environment to initial state
        ## Need to make sure that it is same shape as the observation environment...
        self.state = self.state_init  # self.state needs to be changed for the f()
        observation = np.zeros((84,))
        observation[:4] = [self.state_init[0], self.state_init[1], self.state_init[2], self.state_init[3]]
        observation = np.array(observation, dtype=np.float32)
        # 2. Reset variables of environment to initial values
        self.timestep = 0
        self.cum_reward = 0
        self.done = 0
        # plot variables
        self.dry_weight_plot = []
        self.indoor_co2_plot = []
        self.temp_plot = []
        self.rh_plot = []
        self.supply_co2_plot = []
        self.vent_plot = []
        self.supply_energy_plot = []
        self.timestep_plot = []

        # Weight variable
        self.weight_change = 0
        # 3. Return first observation

        return observation

    # Function to check terminal state:
    def terminal_state(self):
        if self.N == self.timestep:
            self.done = True
            self.printer()
        else:
            self.done = False
            self.timestep += 1
        return self.done

    def close(self):
        return

    ## Adding in the Policy Function HERE...
    def policy_function(self, obs, action):
        """
        DETERMINISTIC...

        This policy needs to go based on the current state variables

        Then a change will be made to an action

        Args:
            - state -> the state (new) 
            - action -> action (normalized)

        Return:
            - Best action

        Action: (Only idx 1,2)
            -	Ventilation rate [mm/s]
            -	Energy supply by heating the system [W/m2]
        
        So essentially our policy will change the temperature if it is outside the threshold

        Target Temperature Range 65F to 70 F -> Convert or find celcius range 

        # So if the current temperture is larger than this then it will decrease the temperature 

        """
        ## Our Policy is Based on Temperature....

        # If temperature is higher than the threshold range
        ## First we reduce heating (this in turn reduces a lot of excess energy usage)
        ## Then we increase the ventilation

        # If temperature is lower than the threshold range
        ## First decrease the ventilation (cheaper than increasing the heating)
        ## Then increase heating

        #### PUT BOUNDS SO DONT GO OUTSIDE ACTION SPACE....
        low_th = 15.00  # C
        high_th = 25.0  # C

        #### PLACED BOUNDS SO THAT THE ACTIONS WERE NOT INCREASED

        if obs[2] < low_th:
            ## This means it is outside the lower bound and we need to increase temperature...
            ### decrease the ventilation,
            if action[1] >= 0.25:
                action[1] -= 0.25
            else:
                action[1] = 0.0
                ### increase the energy for heating
            if action[2] <= 0.75:
                action[2] += 0.25
            else:
                action[2] = 1.0
        elif obs[2] >= high_th:
            ## This means it is outside the upper bound and we need to decrease temperature...
            ### increase the ventilation,
            if action[1] <= 0.75:
                action[1] += 0.25
            else:
                action[1] = 1.0
            ### decrease the energy for heating
            if action[2] >= 0.25:
                action[2] -= 0.25
            else:
                action[2] = 0.0

        # this is a normalized action and is what will be inputted into the step function...

        return action

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
        k2 = self.F(self.state + self.h / 2 * k1, action, d, self.p)
        k3 = self.F(self.state + self.h / 2 * k2, action, d, self.p)
        k4 = self.F(self.state + self.h * k3, action, d, self.p)
        self.state += self.h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return self.state

    def g(self):
        """
        Function that provides the measurements of the environment.
        Give a more readable output of the state variables.

        Returns
            y   --  measurements of the environment
        """
        y = np.array([1e3 * self.state[0],
                      1e-3 * co2dens2ppm(self.state[2], self.state[1]),
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
        ki = np.array([
            p["alfaBeta"] * (
                    (1 - np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
                    (-p["photCO2_1"] * x[2] ** 2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])
                    / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2] ** 2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (
                        x[1] - p["photGamma"])))
            - p["Wc_a"] * x[0] * 2 ** (0.1 * x[2] - 2.5)
            ,

            1 / p["CO2cap"] * (
                    -((1 - np.exp(-p["laiW"] * x[0])) * p["photI0"] * d[0] *
                      (-p["photCO2_1"] * x[2] ** 2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (x[1] - p["photGamma"])
                      / (p["photI0"] * d[0] + (-p["photCO2_1"] * x[2] ** 2 + p["photCO2_2"] * x[2] - p["photCO2_3"]) * (
                                        x[1] - p["photGamma"])))
                    + p["CO2c_a"] * x[0] * 2 ** (0.1 * x[2] - 2.5) + u[0] / 1e6 - (u[1] / 1e3 + p["leak"]) * (
                                x[1] - d[1])
            ),

            1 / p["aCap"] * (
                    u[2] - (p["ventCap"] * u[1] / 1e3 + p["trans_g_o"]) * (x[2] - d[2]) + p["rad_o_g"] * d[0]
            ),

            1 / p["H2Ocap"] * (
                        (1 - np.exp(-p["laiW"] * x[0])) * p["evap_c_a"] * (p["satH2O1"] / (p["R"] * (x[2] + p["T"])) *
                                                                           np.exp(p["satH2O2"] * x[2] / (
                                                                                       x[2] + p["satH2O3"])) - x[3]) - (
                                    u[1] / 1e3 + p["leak"]) * (x[3] - d[3]))]
        )
        return ki

    def printer(self):
        print("Final weight change: " + str(self.weight_change))
        plt.figure(figsize=(15, 15))

        ax_1 = plt.subplot(4, 2, 1)
        plt.plot(self.timestep_plot, self.supply_co2_plot)
        # ax_1.set_title('')
        ax_1.set_xlabel('Time in 15 min steps')
        ax_1.set_ylabel('Supply rate of carbon dioxide [mg]/[m^2][s]')

        ax_2 = plt.subplot(4, 2, 2)
        plt.plot(self.timestep_plot, self.vent_plot)
        ax_2.set_xlabel('Time in 15 min steps')
        ax_2.set_ylabel('Ventilation rate [mm]/[s]')

        ax_3 = plt.subplot(4, 2, 3)
        plt.plot(self.timestep_plot, self.supply_energy_plot)
        ax_3.set_xlabel('Time in 15 min steps')
        ax_3.set_ylabel('Energy supply by heating the system [W]/[m^2]')

        ax_4 = plt.subplot(4, 2, 4)
        plt.plot(self.timestep_plot, self.dry_weight_plot)
        ax_4.set_xlabel('Time in 15 min steps')
        ax_4.set_ylabel('Lettuce dry weight [g]/[m^2]')

        ax_5 = plt.subplot(4, 2, 5)
        plt.plot(self.timestep_plot, self.indoor_co2_plot)
        ax_5.set_xlabel('Time in 15 min steps')
        ax_5.set_ylabel('Indoor CO¬2 concentration [ppm]')

        ax_6 = plt.subplot(4, 2, 6)
        plt.plot(self.timestep_plot, self.temp_plot)
        ax_6.set_xlabel('Time in 15 min steps')
        ax_6.set_ylabel('Indoor air temperature [C]')
        ax_7 = plt.subplot(4, 2, 7)
        plt.plot(self.timestep_plot, self.rh_plot)
        ax_7.set_xlabel('Time in 15 min steps')
        ax_7.set_ylabel('Indoor relative humidity [%]')

        plt.show()
