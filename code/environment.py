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
import copy
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
        nDays= 40,              # simulation days
        Np=20,                # number of future predictions (20 == 5hrs)
        startDay= 150,          # start day of simulation random between 0 and 300..
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
        self.h = h # sampling period, the data is taken every 15 minutes
        self.c = c
        self.nDays = nDays
        self.L = nDays*c # two simulation days in seconds
        self.N = self.L//self.h ## 192 steps

        # action and observation spaces
        ## action space
        ###continuous range of actions
        ### upper and lower values are specified as -1 and 1
        ### actions are represented as array of size of control inputs
        ### Normalized Action Space
        ##  # -	Supply rate of carbon dioxide [mg/m2/s]
        ##  # -	Ventilation rate [mm/s]
        ##  # -	Energy supply by heating the system [W/m2]

        self.action_space = spaces.Box(low=-1*np.ones(nu, dtype=np.float32), high=np.ones(nu, dtype=np.float32))

        ## state space
        ### continuous space given with no upper or lower bounds
        ### shape is of ny+nd*Np 
        #### state space will contain a concatenation of ny greenhouse measurement variables 
        #### and nd*Np weather variables.
        #### Initial Four Measurements... (Current of State Variables)
        #### Then we have Future Predictions for the Four State Variables... (Future Prediction of State Variables)
        #### The observation space is then split up 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(ny + nd*Np,))

        # lower and upper bounds on observations
        self.obs_low = np.array([0., 0.,0, 0.], dtype=np.float32)
        ## first changed max temp [2] from 20 to 30 since lettuce can grow at max of around 29C
        ## but then we changed the bounds to be 0 to +inf for the states
        ### ORIGINAL STATES: 7, 1.6, 30, 70
        ### CHANGED TO....
        self.obs_high = np.array([7.,1.6, 30, 70], dtype=np.float32) # changed max temp to be 30 C cause lettuce can grow at a max of around 29 C

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
 
        ## the done state -> for when program is finished
        ### initialize to False
        self.done =  False
        ## intitialize the net profit sum
        self.net_prof_cum = 0 
        ### measurement inividual values
        measurement = self.g()
        self.dryweight = measurement[0]
        self.indoorco2 = measurement[1]
        self.indoortemp = measurement[2]
        self.indoorrh = measurement[3]
        ## State Old
        self.old_state = np.array([self.dryweight,self.indoorco2, self.indoortemp,self.indoorrh], dtype=np.float32)

        #plot variables
        self.dry_weight_plot =[]
        self.indoor_co2_plot = []
        self.temp_plot = []
        self.rh_plot = []
        self.supply_co2_plot = []
        self.vent_plot = []
        self.supply_energy_plot = []
        self.timestep_plot = []
        self.info = {}
        #Weight variable
        self.weight_change = 0

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
        #print("Initial ACtion:", action)
        action_denorm = (1+action)*(self.max_action - self.min_action)/(2 + self.min_action)

        #print("Action:", action_denorm)
        # 2. Transition state to next state given action and observe environment
        ## obs = next_state
        
        obs = self.f(action_denorm, self.d[self.timestep])
 
        #print("State:",obs)
        #obs = np.clip(obs, self.min_action, self.max_action)

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

        self.dryweight = measurement[0]
        self.indoorco2 = measurement[1]
        self.indoortemp = measurement[2]
        self.indoorrh = measurement[3]

        # 3. Compute reward from profit of greenhouse
        ## how good action was....
        ## can determine where we want to focus...
        ### Ex: focus on minimizing heating and environmental cost...
        ### Ex: focus on the production of lettuce...
        ### The function will then steer what the algorithm will focus on...

       
        
        reward = self.reward_function(obs, action_denorm)

        # 4. return observation, reward, done, info
        # return obs , reward, done, {}
        ### dont need to worry about info it can just be an empty dictionary

        if obs[0] < 0 :
            self.done = True
            reward = 0

        # 5. Check whether state is terminal
        ## how do we know if it is a terminal state... based on if end of simulation so if it has been 2days...
        ## so we will just add one to the timestep since there are 192 periods that we are sampling from
        
        ## First see if self.done has been set to True 
        if self.done != True: 
            # if it hasnt then check if it is true based on terminal state..
            self.done = self.terminal_state()
        


        ## Here we need to add in the environmental data to the observation....
        ### for loop 20 times..

        self.old_state = np.array([self.dryweight,self.indoorco2, self.indoortemp,self.indoorrh], dtype=np.float32)

        for i in range(1,21):# 1 to 20...
            obs = np.concatenate((obs, self.d[self.timestep+i]))
        observation = np.array(obs , dtype=np.float32)

        # observation  = np.zeros((84,))
        # observation[:4] = [obs[0],obs[1],obs[2],obs[3]]
        # observation = np.array(observation , dtype=np.float32)

        # Now we will add our data that we will be plotting to info
        ## info is a dictionary that is able to be accessed locally...
        self.info["timestep_plot"] = self.timestep_plot
        self.info['dry_weight'] =self.dry_weight_plot
        self.info['indoor_co2'] = self.indoor_co2_plot
        self.info['temp'] = self.temp_plot
        self.info['rh'] = self.rh_plot
        self.info["supply_co2"] = self.supply_co2_plot 
        self.info['vent_plot'] = self.vent_plot
        self.info['supply_energy'] = self.supply_energy_plot
        self.info['net_profit'] = self.net_prof_cum 

        #print("--------",)
        return observation, reward, self.done, self.info

    def reward_function(self, obs, action):


        """
        This function will compute the reward for the greenhouse using the equations from

        Reinforcement Learning Versus Model Predictive Control on Greenhouse Climate Control

        Notes:
        In the paper the vector u refers to the actions, and y to the states...

        The paper also defines a time varying bound but this was not included for the sake of simplicitiy...

        So the overall reward function is as follows:

        reward =  r_rev + r_CO2 + r_T - r_controls

        r_rev = self.p["productPrice2"]*abs(self.old_state[0] - obs[0]))
                - the first section computes the price of lettuce based on change in dry weight....so how much revenue we will get

            r_CO2 -> Reward for CO2... will reward  if controlled satisfactorly within given range, and penalty for if outside range...

            if obs[1] < self.obs_low[1]:
                r_CO2 = -cr_co2_1*(obs[1]-self.obs_low[1])**2

            elif obs[1] > self.obs_high[1]: 
                r_CO2 = -cr_co2_1*(obs[1]-self.obs_high[1])**2
            else:
                r_CO2 = cr_co2_2 
            

            r_T -> Reward for temerature to also keep it within a specified range...

            if obs[2] < self.obs_low[2]:
                r_T = -cr_t_1 *(obs[2]-self.obs_low[2])**2

            elif obs[2] > self.obs_high[2]: 
                r_T = -cr_t_1 *(obs[2]-self.obs_high[2])**2
            else:
                r_T = cr_t_2


            ## now for the last bit of the reward we need to compute the affect of tuning the controls..
            #r_controls = r_supp_co2 + r_vent + r_heat
            r_supp_co2 = *action[0]


    
        """
        # additional constants:
        ## CO2 -> constants UNCHANGED!
        cr_co2_1 = .1
        cr_co2_2 = .0005  
        ## Temperature Constants...
        cr_t_1 = .001 *15# changed this so we give more penalty for temp out of range....
        cr_t_2 = .0005

        ## Parameters for the Control
        cr_u1 = -4.5360e-4
        cr_u2 = -.0075 # -.0075 to -.006
        cr_u3 = (-8.5725e-4) *110

        # Constant for the lettuce revenue
        c_r1 = 1 #4 # originally in 16... but tuned in order to get better values... 1 was too low and then 16 was too high then we got 4...

        r_rev = c_r1*(self.dryweight - self.old_state[0])
        ## the first section computes the price of lettuce based on change in dry weight....so how much revenue we will get
        # r_CO2 -> Reward for CO2... will reward  if controlled satisfactorly within given range, and penalty for if outside range...

        if self.indoorco2 < .4: # 
            r_CO2 = -cr_co2_1*(self.indoorco2-.4)**2

        elif self.indoorco2 > .9: 
            r_CO2 = -cr_co2_1*(self.indoorco2-.9)**2
        else:
            r_CO2 = cr_co2_2 
    
        # r_T -> Reward for temerature to also keep it within a specified range...

        if self.indoortemp < 5: # 5 C
            r_T = -cr_t_1 *(self.indoortemp-5)**2

        elif self.indoortemp > 30:  # 30C
            r_T = -cr_t_1 *(self.indoortemp-30)**2
        else:
            r_T = cr_t_2

        ## now for the last bit of the reward we need to compute the affect of tuning the controls..
        #r_controls = r_supp_co2 + r_vent + r_heat
        r_supp_co2 = cr_u1*action[0]
        r_vent = cr_u2*action[1]
        r_heat = cr_u3*action[2]
        r_controls = r_supp_co2 + r_vent + r_heat

        # Now finalize the reward
        reward =  r_rev + r_CO2 + r_T + r_controls



        ## Now compute the net profit... As a single value
        ## First compute the CO2 cost...
        co2_units = 1/(1000*1000) # convert action to kg and divide by the amount of time elapsed in the timestep (seconds)
        cost_CO2 = self.p["co2Cost"] * action[0]*(co2_units)*self.h # euro/m^2 Cost CO2
        ## Ventilation Cost
        tot_vent = (action[1]/1000) + self.p["leak"]
        #### Now our final cost equation related to energy expenditure for ventilation is as follows:
        #### The Ventilation Capacity[J m^{-3}°C^{-1}]  * (Total Ventilation Rate (I/O) [m/s])* Cost of Energy [euro/J] *Indoor Air Temp (Current) [°C] * timestep (s)
        cost_vent = self.p["ventCap"]*tot_vent*self.p["energyCost"]*obs[2]*self.h #euro/m^2 Cost of Energy Related to Ventilation

        ### The cost of heating
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [W/m^2] (Convert from W to Joule/s)
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [J/(s*(m^2))] * timestep (seconds)
        heat_cost =  self.p["energyCost"] * action[2] * self.h ## [e]uro/m^2]

        
        ### Total cost of energy [euro/m^2]
        total_cost_energy = heat_cost + cost_vent

        ## total expenses [euro/m^2]
        total_expenses = total_cost_energy + cost_CO2


        if self.timestep == 0:
            total_revenue = ( self.dryweight/1000 - self.old_state[0]/1000)*self.p["productPrice2"] #+ self.p["productPrice1"] # Auction Price of Lettuce euro/m^2
        else:
            # dont add the extra..
            total_revenue = (self.dryweight/1000 - self.old_state[0]/1000 )*self.p["productPrice2"]   # Auction Price of Lettuce euro/m^2

        net_profit = float((total_revenue) - (total_expenses))*100
     

        self.net_prof_cum += net_profit

        self.weight_change += (self.dryweight-self.old_state[0])

        return reward

    

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
        ## Need to make sure that it is same shape as the observation environment...
        self.state = self.state_init # self.state needs to be changed for the f()
        ### measurement inividual values
        measurement = self.g()
        self.dryweight = measurement[0]
        self.indoorco2 = measurement[1]
        self.indoortemp = measurement[2]
        self.indoorrh = measurement[3]

        obs = copy.deepcopy(self.state)
        # 2. Reset variables of environment to initial values
        self.timestep = 0
        self.net_prof_cum = 0 
        self.done = 0
        #plot variables
        self.dry_weight_plot =[]
        self.indoor_co2_plot = []
        self.temp_plot = []
        self.rh_plot = []
        self.supply_co2_plot = []
        self.vent_plot = []
        self.supply_energy_plot = []
        self.timestep_plot = []
        self.info = {}
        #Weight variable
        self.weight_change = 0
        for i in range(1,21):
            obs = np.concatenate((obs, self.d[self.timestep+i]))
        observation = np.array(obs , dtype=np.float32)
        # 3. Return first observation


        return observation
    

    # Function to check terminal state:
    def terminal_state(self):
        if self.N == self.timestep:
            self.done = True
            #self.printer()
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
        low_th = 13.50 # C
        high_th = 16.50 # C

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
