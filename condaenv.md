# Create Duplicate Conda Environment 
Instead of manually installing the imporant packages create a duplicate conda environment based on the envrionment.yml file provided

This yml file contains all the necessary packages in order to run the files outlined in the Getting Started Markdown File. Please follow the below steps in order to properly transfer this yml file to your machine as a new conda environment. 

1. Enter a Shell or Bash Terminal 

2. While in the shell or bash terminal 

Deactivate your current conda environment in order to avoid any possible conflicts.
However, if it is a new terminal window and there is no active environment then
this step is not necessary.

` conda deactivate `


3. After ensuring that you are in a NON Anaconda Prompt terminal with NO active anaconda environment
then proceeed to the directory where the environment.yml file is located.



4. Once the environment.yml file has been located then run the following command:

Before creating the environment either make this environment have a custom name, or ensure that Lettuce_GH_RL is not already an anaconda environment.

`conda env create -f environment.yml `

This process may take a while upwards of 5-10 minutes

5. After you have your new conda environment activate it
We have provided the environment a name for you in the yml file so use this name as provided unless you have initialized it when creating with a different name.

`conda activate Lettuce_GH_RL`

6. Before proceeding further please run the `run_env_disc.ipynb` notebook to determine whether or not your current environment is able to run the gym environments, and generate the matplotlib plots. If these plots are not able to be generated then please go to your anaconda prompt and activate your current conda environment and uninstall matplotlib and then reinstall.

`pip uninstall matplotlib`

then

`pip install matplotlib`

7. Now go back and run the `run_env_disc.ipynb` notebook and there should not be anymore errors. However, if there are then please see the **Important** section below.

8. Now that you have an activate environment that is working please proceed back to the README markdown file to see what is available within this project.


**Important: IF ENCOUNTERING ERRORS**

1. Make sure to install the vs studio C++ components for testing openai environments
2. If there are issues with the installation of stable baselines 3, then attempt the documentation in the installations markdown file within the `other` directory. 




