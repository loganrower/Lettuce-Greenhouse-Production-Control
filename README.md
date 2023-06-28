# Lettuce Greenhouse Production Control
Project for Advanced Machine Learning at Wageningen University 

Authors of the Project: Bart van Laatum; Congcong Sun 

Project Group Members: Eric Wiskandt and Logan Rower

Greenhouse model is derived from (Boersma & van Mourik, 2021). 

Objective:
- Utilize Reinforcement Learning Algorithms to control the indoor environmental conditions of a lettuce greenhouse with the goal of maximizing net profit.


Boersma S, van Mourik S. 2021. Nonlinear sample-based MPC in a Greenhouse with Lettuce and uncertain weather forecasts. 40th Benelux Workshop on Systems and Control, 58-59.


Before Proceeding with this project:
   1. If wanting to run this on your local machine see the installations markdown file and install the required dependancies. 
   Follow the requirements exactly.
   2. However if you are wanting to run this project within google colab you should follow the Lettuce_GH_RL.ipynb jupyter notebook. Either download this and clone the github repository or use some other method in order to to load this into google colab.
   3. The Lettuce_GH_RL.ipynb should be developed but if it is still in development continue with the installation on your local machine.


Ways to utilize this project:
- Deterministic:
    1. For testing a deterministic approach with the greenhouse simulated environment utilize the 
- Training a new model:
    1. open the environment.py file and preset your environmental parameters and whether you want to try to use the preset 
    reward function or the net profit as the reward function.
    2. train your model based on your simulation time and the
    3. 

Documentation for how to utilize this project:


- See bestmodels.md for the trained models that can be run in the gh_eval.py. 