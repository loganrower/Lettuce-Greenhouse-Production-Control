def reward_function(self, obs, action):
    # additional constants:
    ## CO2 -> constants
    cr_co2_1 = .1
    cr_co2_2 =  .0005  
    ## Temperature Constants...
    cr_t_1 = .001
    cr_t_2 = .0005

    ## Parameters for the Control
    cr_u1 = -4.5360e-4
    cr_u2 = -.0075
    cr_u3 = -8.5725e-4

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

    r_rev = self.p["productPrice2"]*abs(self.old_state[0] - obs[0])
    ## the first section computes the price of lettuce based on change in dry weight....so how much revenue we will get

    # r_CO2 -> Reward for CO2... will reward  if controlled satisfactorly within given range, and penalty for if outside range...

    if obs[1] < self.obs_low[1]:
        r_CO2 = -cr_co2_1*(obs[1]-self.obs_low[1])**2

    elif obs[1] > self.obs_high[1]: 
        r_CO2 = -cr_co2_1*(obs[1]-self.obs_high[1])**2
    else:
        r_CO2 = cr_co2_2 
    

    # r_T -> Reward for temerature to also keep it within a specified range...

    if obs[2] < self.obs_low[2]:
        r_T = -cr_t_1 *(obs[2]-self.obs_low[2])**2

    elif obs[2] > self.obs_high[2]: 
        r_T = -cr_t_1 *(obs[2]-self.obs_high[2])**2
    else:
        r_T = cr_t_2


    ## now for the last bit of the reward we need to compute the affect of tuning the controls..
    #r_controls = r_supp_co2 + r_vent + r_heat
    r_supp_co2 = cr_u1*action[0]
    r_vent = cr_u2*action[1]
    r_heat = cr_u3*action[2]
    r_controls = r_supp_co2 + r_vent + r_heat

    # Now finalize the reward

    reward =  r_rev + r_CO2 + r_T - r_controls

    return reward





    



# OTHER;


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

        #TODO: implement reward function.
        # Main goals of this functions are to:
        # 1. Compute reward of greenhouse (e.g., profit of the greenhouse)
        ## first compute the total expense

        ### cost of CO2 (CO2 added)
        ##### cost of CO2 = CO2_Cost [€ kg^{-1}{CO2}] *  Supply Rate of CO2[mg/m2*s] * s *(1g/1000mg) *(1kg/1000g)
        #    ### What about using the CO2 Supply Rate.... This is more with respect to the cost to supply CO2...
        #    ### What about amount observed indoors as apart of the state? Amount of CO2 Observed Indoors (state[1])[kg/m3]
        #    ### What about CO2_Capacity [m^3{air} m^{-2}{gh}]
        co2_units = 1/(1000*1000) # convert action to kg and divide by the amount of time elapsed in the timestep (seconds)
        cost_CO2 = self.p["co2Cost"] * action[0]*(co2_units)*self.h # euro/m^2 Cost CO2
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
        tot_vent = (action[1]/1000) + self.p["leak"]
        #### Now our final cost equation related to energy expenditure for ventilation is as follows:
        #### The Ventilation Capacity[J m^{-3}°C^{-1}]  * (Total Ventilation Rate (I/O) [m/s])* Cost of Energy [euro/J] *Indoor Air Temp (Current) [°C] * timestep (s)
        cost_vent = self.p["ventCap"]*tot_vent*self.p["energyCost"]*obs[2]*self.h #euro/m^2 Cost of Energy Related to Ventilation
        
        ### The cost of heating
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [W/m^2] (Convert from W to Joule/s)
        #### heat_cost =  cost of energy [euro/J] * Energy Supply by heating the system [J/(s*(m^2))] * timestep (seconds)
        heat_cost =  self.p["energyCost"] * action[2] * self.h ## [e]uro/m^2]


        ## increase dryweight...
        r_k = change_in_dry_weight

        
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
            total_revenue = (abs(self.old_state[0] - obs[0])*self.p["productPrice2"] ) #+ self.p["productPrice1"] # Auction Price of Lettuce euro/m^2
        else:
            # dont add the extra..
            total_revenue = (abs(self.old_state[0] - obs[0]))*self.p["productPrice2"]   # Auction Price of Lettuce euro/m^2

        # 2. return reward|
        net_profit = float((total_revenue) - (total_expenses))
        #print("timestep:",self.timestep)
        #print("Total Rev", total_revenue)
        #print("Total Expenses",total_expenses)
        #print("Weight change: " + str((obs[0]-self.old_state[0])*1000))
        self.weight_change += (obs[0]-self.old_state[0])*1000
        #print("Net Profit", net_profit)
        #print("Cumulative Reward", self.cum_reward)
        # self.cum_reward += net_profit

        

        self.net_profit_plot.append(net_profit)

        reward = net_profit*100

    #PENALIZE OUT OF BOUNDS! GOT RID OF THIS...
        for idx, obs_val in enumerate(obs):
            obs_high_val = self.obs_high[idx] 
            obs_low_val = self.obs_low[idx]
            if obs_val < 0 or obs_val > (obs_high_val+10): 
                print("time:", self.timestep )
                print("state outside range:", obs_val, idx, (obs_low_val,obs_high_val ))
        #         # go to end state
                #self.done = True
                reward -= 1


        # Was getting lots of potential dropout states from high temp... so set this so we can avoid neg profit..
        #reward = net_profit*100
        # if net_profit < 0:
        #     # then self.done = True
            #  self.done = True


        # # reward = net_profit *100
        if net_profit > 0:
             reward += 10
        # else:
        #     reward = -.1
        
        # # checked to see if we reached the weight....
        # ## deterministic gets to total of 203 so we do 10% of that...
        # if self.weight_change >= .203 : # achieved for starting day 90, 20 day period we set the target weight change..... to reach is.... kg/m^2 
        #     reward += 100      

        # checkled to see if we reached the profit...
        # if net_profit >= :
        #     reward = 

        # net_profit = net_profit*100
        
        # Append net profit to the net profit list..
        

        return reward