# Create Duplicate Conda Environment 
Instead of manually installing the imporant packages create a duplicate conda environment based on the envrionment.yml file provided

This yml file contains all the necessary packages in order to run the files outlined in the Getting Started Markdown File. Please follow the below steps in order to properly transfer this yml file to your machine as a new conda environment. 

1. Deactivate your current conda environment in order to avoid any possible conflicts.

2. Run the following command:

` conda env create -f environment.yml `

This process may take a while upwards of 5 minutes

3. After you have your new conda environment activate it
We have provided the environment a name for you in the yml file so use this name as provided unless you have initialized it when creating with a different name.

`conda activate Lettuce_GH_RL`

4. Now that you have an activate environment please proceed back to the ReadME markdown file to see what is available within this project.



