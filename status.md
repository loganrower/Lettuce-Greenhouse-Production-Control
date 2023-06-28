# Project Status

**Assignment 1:** `COMPLETETED`

    - step() -> Done
    - reward() -> Done
        - Reward function seems to work
        - Dimensional Analysis of our thought process should be written out in latex, with a detailed explanation.
    - terminal_state() -> Done
    - Visualization -> NOT DONE
        - using the info variable...

**Assignment 2:** `COMPLETED`

    - Implement the reset() function -> Done

**Assignment 3:** -> `IN-PROGRESS`

    - Deterministic Policy -> IN PROGRESS
        - review the policy if time 
        - Need to make sure that the mutated values do not exceed the min,max bounds.

    - Visualise Trajectories

    NOTES:
        - Currently having issues with the temperature displayed but it might just be due to the fact it is a normalized value, but need to check this

        - Make nice plots using the function g(), and also based on the actual action and state parameters as well. It might be better to use g() for the state parameters though so we can check this.

        - Needed to change the start date from 40 to 90 in order to start end of march, and went over 40 days in order to capture more data...
            - Since we are only using natural lighting so we want to plant in spring and that is more realistic..
        
        - Errors stemming from large or really small states
            - Changed the obsevations bounds from -inf and +inf to the obs low and obs high


            - then i also used clip since the issue with PPO is with the state space since it acccounts for the action space in the algorithm but we need to enforce the state space range ourselves... using np.clip()

            - ISSUE RESOLVED
                -  and was solved by essentially redoing the denormalization equation since our actions were becoming negative....
                - so now we dont have negative actions and they are in the appropriate range!
    

Assigment Step 4:
    - Model currently running with PPO, and most recent saved model is really good, and works well. It has positive reward, and this was after implementing the changes in order to fix our actions from being negative...   

    - So the model that is first in the models folder 1687177848 -> PPO, this model was run for 100k timesteps, training every 10000. We see in our model that the episode mean reward continued to increase, and increase....

    ISSUE:
    - The two models we see that have run have both died at some point due to some sort of nan issue... not sure where or why this occurred... Essentially current state changes such that it has nan values..... but the denormalized actions anr in proper rnage and same with the old_state...

    - So why does this occur?