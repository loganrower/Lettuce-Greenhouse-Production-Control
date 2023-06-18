## Python Package Local Installation Instructions

1. Make a new anaconda environment Python 10.x.x

2. First install swig...
3. First install the version of gym that will be used
    pip3 install git+https://github.com/Pythoniasm/gym-fork.git@fix-v0.21.0

4. Next will then install stable baselines3 version
    - clone the repository of stablebaselines
        git clone https://github.com/DLR-RM/stable-baselines3

    - then we will go into the cloned stablebaselines directory and reset the version of the repo to v1.5.0
        cd stable-baselines3 && git reset --hard v1.5.0

    - Then went into the setup.py file in this repo and changed the following...
        Manually edit the file and make the desired changes. In this case, you would replace 0.21 with 0.21.1 and torch>=1.8.1 with torch==1.11.0
    
            - CURRENTLY IT IS SET SO IT IS torch>=1.11.0.... hopefully this works if not need to check...
    - now we will install the package while within the directory stablebaseline3 still
         pip3 install -e .

    - install gymnasium
        pip install gymnasium
5. Make sure to install the vs studio C++ components for testing openai environments

6. For windows use the following...

conda install -c conda-forge pyvirtualdisplay
conda install -c conda-forge xorg-libx11

 virtual display implementation similar to xvfb, and xorg-libx11 is a library required for X Window System compatibility.

    
7. Install gym[box2D] for practice...
    - using the older gym instead of gymnasium.....
    pip install gym[box2D]

8. In order to get a visual for lets say the lunar lander we need to update the installation of pyglet...
    https://stackoverflow.com/questions/74314778/nameerror-name-glpushmatrix-is-not-defined 
    pip install pyglet==1.5.27

9. get tensoboard working...

go to bash, make sure that activating conda environment

then go to directory where log files are..

then run following:

tensorboard --logdir ./logs/
